import re
import json
import uuid
import logging
from typing import Dict, Any, List, Optional
from rapidfuzz import process, fuzz
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.db.models import ConceptMastery
from app.llm.gateway import call_llm_gateway
from app.llm.prompts import CURRICULUM_GRAPH_PROMPT
from app.llm.schemas import ConceptGraphSchema

logger = logging.getLogger(__name__)

def slugify_topic(topic: str) -> str:
    """Normalizes topic input string into clean lowercase hyphenated slug."""
    s = topic.strip().lower()
    s = re.sub(r'[^a-z0-9]+', '-', s)
    return s.strip('-')

async def get_or_create_concept_graph(
    topic: str,
    mongo_db: Any,
    target_level: str = "expert",
    knowledge_profile: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Retrieves existing concept_graphs document from MongoDB or generates a new one with cycle validation."""
    slug = slugify_topic(topic)
    collection = mongo_db.get_collection("concept_graphs")
    
    # 1. Exact match by slug
    existing = await collection.find_one({"_id": slug})
    if existing:
        return existing

    # 2. Fuzzy match against topic aliases
    all_graphs = await collection.find({}).to_list(100)
    for g in all_graphs:
        aliases = g.get("aliases", []) + [g.get("topic", ""), g.get("_id", "")]
        match = process.extractOne(topic, aliases, scorer=fuzz.token_sort_ratio)
        if match and match[1] >= 85:
            logger.info(f"Fuzzy matched '{topic}' to existing concept graph '{g['_id']}'")
            return g

    # 3. Generate NEW concept graph using LangChain single-shot LLM call
    profile_str = json.dumps(knowledge_profile) if knowledge_profile else "None"
    res = await call_llm_gateway(
        CURRICULUM_GRAPH_PROMPT,
        {
            "topic": topic,
            "target_level": target_level,
            "knowledge_profile": profile_str
        },
        ConceptGraphSchema
    )

    raw_nodes = res.nodes or res.concepts or []
    concepts_data = [c.model_dump() if hasattr(c, "model_dump") else c for c in raw_nodes]
    
    for c in concepts_data:
        cid = c.get("concept_id") or c.get("id")
        c["id"] = cid
        c["concept_id"] = cid

    # Topological cycle validation
    validate_topological_sort(concepts_data)

    doc = {
        "_id": slug,
        "topic": res.topic,
        "target_level": getattr(res, "target_level", target_level) or target_level,
        "aliases": res.aliases,
        "concepts": concepts_data,
        "nodes": concepts_data
    }
    await collection.insert_one(doc)
    return doc

def validate_topological_sort(concepts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Cleans dependencies, removes phantom/self references, 
    and handles cycles gracefully without hanging the server.
    """
    valid_ids = {
        (c.get("concept_id") or c.get("id"))
        for c in concepts
        if (c.get("concept_id") or c.get("id"))
    }
    
    # 1. Sanitize: remove dependencies that don't exist in the concept list or self-references
    for c in concepts:
        cid = c.get("concept_id") or c.get("id")
        cleaned_deps = [
            dep for dep in c.get("depends_on", [])
            if dep in valid_ids and dep != cid
        ]
        c["depends_on"] = cleaned_deps
        if "id" not in c and cid:
            c["id"] = cid
        if "concept_id" not in c and cid:
            c["concept_id"] = cid

    # 2. Kahn's Algorithm
    in_degree = {(c.get("concept_id") or c["id"]): 0 for c in concepts}
    adj_list = {(c.get("concept_id") or c["id"]): [] for c in concepts}

    for c in concepts:
        cid = c.get("concept_id") or c["id"]
        for dep in c["depends_on"]:
            adj_list[dep].append(cid)
            in_degree[cid] += 1

    queue = [c_id for c_id, deg in in_degree.items() if deg == 0]
    sorted_order = []

    while queue:
        curr = queue.pop(0)
        sorted_order.append(curr)
        for neighbor in adj_list[curr]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # If cycle exists, break cycles forcefully instead of crashing/hanging
    if len(sorted_order) < len(concepts):
        logger.warning("[validate_topological_sort] Cycle detected from LLM! Pruning cyclic edges...")
        unresolved = [c for c in concepts if (c.get("concept_id") or c["id"]) not in sorted_order]
        for c in unresolved:
            c["depends_on"] = []  # Clear cyclic dependencies safely
            sorted_order.append(c.get("concept_id") or c["id"])

    return concepts

async def generate_personalized_learning_path(
    user_id: uuid.UUID,
    topic: str,
    target_level: str,
    knowledge_profile: Dict[str, Dict[str, Any]],
    diagnostic_session_id: Optional[uuid.UUID],
    mongo_db: Any,
    db: AsyncSession
) -> Dict[str, Any]:
    """Generates personalized learning path, stores document in Mongo, and seeds Postgres concept_mastery rows."""
    concept_graph = await get_or_create_concept_graph(
        topic,
        mongo_db,
        target_level=target_level,
        knowledge_profile=knowledge_profile
    )
    concepts = concept_graph.get("nodes") or concept_graph.get("concepts", [])
    
    # SAFETY CHECK: If the diagnostic session tested specific concepts, ensure
    # we aren't using a mismatched fallback graph of only 5 generic items.
    profile_keys = set(knowledge_profile.keys())
    graph_keys = {(c.get("concept_id") or c.get("id")) for c in concepts if (c.get("concept_id") or c.get("id"))}
    
    # If the graph concepts have zero intersection with the diagnostic session concepts,
    # the cached concept graph is out of sync with the diagnostic session.
    if profile_keys and not graph_keys.intersection(profile_keys):
        logger.warning(
            f"Concept graph keys ({graph_keys}) do not match diagnostic session keys ({profile_keys}). "
            "Re-generating concept graph dynamically from profile keys..."
        )
        # Adapt graph nodes directly from the knowledge profile to maintain consistency
        concepts = [
            {
                "id": cid,
                "concept_id": cid,
                "level": "beginner" if idx < len(profile_keys) // 3 else ("intermediate" if idx < 2 * len(profile_keys) // 3 else "advanced"),
                "concept_type": "conceptual",
                "depends_on": []
            }
            for idx, cid in enumerate(profile_keys)
        ]
        concept_graph["concepts"] = concepts
        concept_graph["nodes"] = concepts

    nodes = []
    skipped_concepts = []

    for c in concepts:
        c_id = c.get("concept_id") or c.get("id")
        prof = knowledge_profile.get(c_id, {"confidence": 0.0, "status": "unknown"})
        conf = prof.get("confidence", 0.0)
        
        status = c.get("status")
        action = c.get("action")
        est = c.get("estimated_minutes")

        if not status or status not in ["known", "partial", "learning", "locked"]:
            if conf >= 0.8:
                status = "known"
                action = "skip_teach_verify_only"
                est = 0
            elif conf >= 0.4:
                status = "partial"
                action = "refresher"
                est = 10
            else:
                has_prereqs = bool(c.get("depends_on"))
                status = "locked" if has_prereqs and not knowledge_profile else "learning"
                action = "full_teach"
                est = 20

        if status == "known":
            skipped_concepts.append(c_id)

        nodes.append({
            "concept_id": c_id,
            "level": c.get("level", "beginner"),
            "concept_type": c.get("concept_type", "conceptual"),
            "status": status,
            "action": action,
            "estimated_minutes": est if est is not None else 15,
            "depends_on": c.get("depends_on", [])
        })

        # Seed Postgres concept_mastery table
        stmt = select(ConceptMastery).where(
            ConceptMastery.user_id == user_id,
            ConceptMastery.topic == concept_graph["_id"],
            ConceptMastery.concept_id == c_id
        )
        res = await db.execute(stmt)
        existing_mastery = res.scalar_one_or_none()

        if not existing_mastery:
            mastery_row = ConceptMastery(
                user_id=user_id,
                topic=concept_graph["_id"],
                concept_id=c_id,
                mastery=conf,
                attempts=0,
                failures=0,
                status=status
            )
            db.add(mastery_row)

    await db.commit()

    # Determine first active concept
    first_active = next((n["concept_id"] for n in nodes if n["status"] != "known"), nodes[0]["concept_id"])

    lp_id = f"lp_{uuid.uuid4().hex[:8]}"
    learning_path_doc = {
        "_id": lp_id,
        "user_id": str(user_id),
        "topic": concept_graph["_id"],
        "target_level": target_level,
        "generated_from_diagnostic_session": str(diagnostic_session_id) if diagnostic_session_id else None,
        "nodes": nodes,
        "current_node": first_active
    }

    lp_collection = mongo_db.get_collection("learning_paths")
    await lp_collection.insert_one(learning_path_doc)

    return learning_path_doc