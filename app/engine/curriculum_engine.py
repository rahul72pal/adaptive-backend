import re
import json
import uuid
import logging
import random
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from rapidfuzz import process, fuzz
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.db.models import ConceptMastery, UserTopic
from app.llm.gateway import call_llm_gateway
from app.llm.prompts import CURRICULUM_GRAPH_PROMPT, LEARNING_CONTENT_PROMPT, DIAGNOSTIC_CHECKLIST_PROMPT
from app.llm.schemas import ConceptGraphSchema, LessonContentSchema, DiagnosticChecklistSchema

logger = logging.getLogger(__name__)

LEVEL_ORDER = ["beginner", "intermediate", "advanced", "expert"]

def slugify_topic(topic: str) -> str:
    """Normalizes topic input string into clean lowercase hyphenated slug."""
    s = topic.strip().lower()
    s = re.sub(r'[^a-z0-9]+', '-', s)
    return s.strip('-')

def normalize_topic(topic: str) -> Dict[str, str]:
    """Normalizes user input string into user_topic, canonical_topic, and slug."""
    user_topic = topic.strip()
    slug = slugify_topic(user_topic)
    canonical_topic = user_topic.replace("-", " ").replace("_", " ").title()
    return {
        "user_topic": user_topic,
        "canonical_topic": canonical_topic,
        "slug": slug
    }


async def get_or_create_user_topic(
    user_id: uuid.UUID,
    raw_topic: str,
    mongo_db: Any
) -> Dict[str, Any]:
    """
    Selects or creates a topic linked to user_id in MongoDB 'user_topics' collection.
    Handles canonical normalization, alias matching, and fuzzy matching (e.g. 'js in Array' -> 'Array').
    Initializes completed_nodes and completed_node_ids to track user's completed concepts.
    """
    from datetime import datetime, timezone
    norm = normalize_topic(raw_topic)
    input_clean = raw_topic.strip()
    input_slug = norm["slug"]
    user_id_str = str(user_id)

    if mongo_db is None:
        return {
            "topic_slug": input_slug,
            "canonical_topic": norm["canonical_topic"],
            "raw_input_topic": input_clean,
            "aliases": [input_slug, input_clean.lower(), norm["canonical_topic"].lower()],
            "completed_nodes": [],
            "completed_node_ids": []
        }

    ut_collection = mongo_db.get_collection("user_topics")
    cg_collection = mongo_db.get_collection("concept_graphs")

    # 1. Exact match by topic_slug or existing aliases in user's saved topics
    existing_ut = await ut_collection.find_one({
        "user_id": user_id_str,
        "$or": [
            {"topic_slug": input_slug},
            {"aliases": input_slug},
            {"aliases": input_clean.lower()}
        ]
    })

    if existing_ut:
        aliases = existing_ut.get("aliases", [])
        updated = False
        update_fields = {}
        if input_clean.lower() not in aliases:
            aliases.append(input_clean.lower())
            update_fields["aliases"] = aliases
            existing_ut["aliases"] = aliases
            updated = True
        if not existing_ut.get("canonical_topic"):
            canon = norm["canonical_topic"]
            update_fields["canonical_topic"] = canon
            existing_ut["canonical_topic"] = canon
            updated = True
        if not existing_ut.get("raw_input_topic"):
            update_fields["raw_input_topic"] = input_clean
            existing_ut["raw_input_topic"] = input_clean
            updated = True
        if not existing_ut.get("topic_slug"):
            update_fields["topic_slug"] = input_slug
            existing_ut["topic_slug"] = input_slug
            updated = True
        if not existing_ut.get("status"):
            update_fields["status"] = "active"
            existing_ut["status"] = "active"
            updated = True
        if "completed_nodes" not in existing_ut:
            update_fields["completed_nodes"] = []
            existing_ut["completed_nodes"] = []
            updated = True
        if "completed_node_ids" not in existing_ut:
            update_fields["completed_node_ids"] = []
            existing_ut["completed_node_ids"] = []
            updated = True
        if updated:
            update_fields["updated_at"] = datetime.now(timezone.utc).isoformat()
            await ut_collection.update_one(
                {"_id": existing_ut["_id"]},
                {"$set": update_fields}
            )
        return existing_ut

    # 2. Fuzzy & alias search across existing concept graphs in Mongo to resolve canonical name
    canonical_topic = norm["canonical_topic"]
    canonical_slug = input_slug
    aliases_list = [input_slug, input_clean.lower(), canonical_topic.lower()]

    try:
        all_graphs = await cg_collection.find({}).to_list(200)
        best_match_slug = None
        best_match_canonical = None

        for g in all_graphs:
            g_topic = g.get("canonical_topic") or g.get("topic", "")
            g_aliases = g.get("aliases", []) + [g.get("topic", ""), g_topic]
            match = process.extractOne(canonical_topic, g_aliases, scorer=fuzz.token_sort_ratio)
            if match and match[1] >= 80:
                best_match_slug = g.get("topic") or slugify_topic(g_topic)
                best_match_canonical = g_topic
                break

        if best_match_slug and best_match_canonical:
            canonical_slug = best_match_slug
            canonical_topic = best_match_canonical
            if input_clean.lower() not in aliases_list:
                aliases_list.append(input_clean.lower())
    except Exception as search_err:
        logger.warning(f"[get_or_create_user_topic] Fuzzy graph lookup error: {search_err}")

    doc_id = f"ut_{user_id_str[:8]}_{canonical_slug}"
    now_iso = datetime.now(timezone.utc).isoformat()

    doc = {
        "_id": doc_id,
        "user_id": user_id_str,
        "topic_slug": canonical_slug,
        "canonical_topic": canonical_topic,
        "raw_input_topic": input_clean,
        "aliases": list(set(aliases_list)),
        "mode_choice": None,
        "diagnostic_session_id": None,
        "current_level": "beginner",
        "completed_nodes": [],
        "completed_node_ids": [],
        "status": "active",
        "created_at": now_iso,
        "updated_at": now_iso
    }

    await ut_collection.update_one(
        {"_id": doc_id},
        {"$set": doc},
        upsert=True
    )
    return doc


async def record_user_completed_node(
    user_id: Any,
    topic_slug: str,
    concept_id: str,
    mongo_db: Any,
    db: Optional[AsyncSession] = None,
    level: Optional[str] = None
) -> Dict[str, Any]:
    """
    Stores details of the completed concept node in user_topics table/collection
    specifically for this user, without updating or corrupting the shared concept graph.
    """
    user_id_str = str(user_id)
    concept_clean = concept_id.strip().lower()
    now_iso = datetime.now(timezone.utc).isoformat()
    node_entry = {
        "concept_id": concept_clean,
        "status": "mastered",
        "level": level or "beginner",
        "completed_at": now_iso
    }

    # 1. Update MongoDB user_topics
    if mongo_db is not None:
        try:
            ut_collection = mongo_db.get_collection("user_topics")
            norm = normalize_topic(topic_slug)
            await ut_collection.update_one(
                {"user_id": user_id_str, "topic_slug": topic_slug},
                {
                    "$addToSet": {"completed_node_ids": concept_clean},
                    "$push": {"completed_nodes": node_entry},
                    "$set": {"updated_at": now_iso},
                    "$setOnInsert": {
                        "canonical_topic": norm["canonical_topic"],
                        "raw_input_topic": norm["user_topic"],
                        "aliases": [topic_slug, norm["canonical_topic"].lower()],
                        "current_level": level or "beginner",
                        "status": "active",
                        "created_at": now_iso
                    }
                },
                upsert=True
            )
        except Exception as e:
            logger.warning(f"[record_user_completed_node] Mongo user_topics update failed: {e}")

    # 2. Update PostgreSQL user_topics
    if db is not None:
        try:
            uid_uuid = uuid.UUID(user_id_str) if isinstance(user_id, str) else user_id
            stmt = select(UserTopic).where(
                UserTopic.user_id == uid_uuid,
                UserTopic.topic_slug == topic_slug
            )
            res = await db.execute(stmt)
            ut_row = res.scalar_one_or_none()
            if not ut_row:
                ut_row = UserTopic(
                    user_id=uid_uuid,
                    topic_slug=topic_slug,
                    canonical_topic=topic_slug.replace("-", " ").replace("_", " ").title(),
                    current_level=level or "beginner",
                    status="active",
                    completed_nodes=[node_entry],
                    completed_node_ids=[concept_clean]
                )
                db.add(ut_row)
            else:
                curr_ids = list(ut_row.completed_node_ids or [])
                if concept_clean not in curr_ids:
                    curr_ids.append(concept_clean)
                    ut_row.completed_node_ids = curr_ids
                    curr_nodes = list(ut_row.completed_nodes or [])
                    curr_nodes.append(node_entry)
                    ut_row.completed_nodes = curr_nodes
            await db.commit()
        except Exception as e:
            logger.warning(f"[record_user_completed_node] Postgres user_topics update failed: {e}")

    return node_entry

def validate_concept_nodes(concepts: List[Dict[str, Any]], level: str = "beginner") -> None:
    """Validate LLM-generated curriculum nodes before persistence."""
    if not concepts:
        raise ValueError(f"LLM generated an empty concept graph for level '{level}'")

    required_fields = [
        "concept_id",
        "level",
        "node_type",
        "concept_type",
        "status",
        "action",
        "depends_on",
    ]

    seen_ids = set()

    for index, concept in enumerate(concepts):
        concept_id = concept.get("concept_id") or concept.get("id")

        if not concept_id or not isinstance(concept_id, str) or not concept_id.strip():
            raise ValueError(
                f"Invalid concept at index {index}: missing or empty concept_id. Node={concept}"
            )

        cid = concept_id.strip()
        concept["concept_id"] = cid
        concept["id"] = cid

        if cid in seen_ids:
            raise ValueError(
                f"Duplicate concept_id generated by LLM: '{cid}'"
            )

        seen_ids.add(cid)

        if "level" not in concept or not concept["level"]:
            concept["level"] = level
        if "node_type" not in concept or concept["node_type"] not in ["learning", "practice"]:
            concept["node_type"] = "learning"
        if "concept_type" not in concept or not concept["concept_type"]:
            concept["concept_type"] = "conceptual"
        if "status" not in concept or not concept["status"]:
            concept["status"] = "learning"
        if "action" not in concept or not concept["action"]:
            concept["action"] = "full_teach"
        if "depends_on" not in concept or concept["depends_on"] is None:
            concept["depends_on"] = []

        missing = [
            field
            for field in required_fields
            if field not in concept
        ]

        if missing:
            raise ValueError(
                f"Concept '{cid}' is missing required fields: {missing}"
            )

        if concept["node_type"] not in ["learning", "practice"]:
            raise ValueError(
                f"Invalid node_type '{concept['node_type']}' for concept '{cid}'"
            )

    # Validate dependencies & remove self-dependencies
    for concept in concepts:
        cid = concept["concept_id"]
        cleaned_deps = []
        for dependency in concept.get("depends_on", []):
            if dependency == cid:
                logger.warning(f"Pruning self-dependency on concept '{cid}'")
                continue
            cleaned_deps.append(dependency)
        concept["depends_on"] = cleaned_deps

def validate_topological_sort(concepts: List[Dict[str, Any]], valid_prereq_ids: Optional[set] = None) -> List[Dict[str, Any]]:
    """
    Cleans dependencies, removes phantom/self references, 
    and handles cycles gracefully without hanging the server.
    """
    valid_ids = {
        (c.get("concept_id") or c.get("id"))
        for c in concepts
        if (c.get("concept_id") or c.get("id"))
    }
    if valid_prereq_ids:
        valid_ids.update(valid_prereq_ids)
    
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
            if dep in adj_list:
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

def get_available_nodes(nodes: List[Dict[str, Any]], completed_ids: Optional[set] = None) -> List[Dict[str, Any]]:
    """
    Finds nodes in the DAG whose prerequisites are all satisfied (completed/known)
    and whose own status is not yet 'known' or 'mastered'.
    """
    if completed_ids is None:
        completed_ids = {
            n.get("concept_id") or n.get("id")
            for n in nodes
            if n.get("status") in ["known", "mastered"]
        }

    available = []
    for node in nodes:
        cid = node.get("concept_id") or node.get("id")
        if node.get("status") in ["known", "mastered"]:
            continue
        deps = set(node.get("depends_on", []))
        if deps.issubset(completed_ids):
            available.append(node)
    return available

async def get_previous_concept_ids(
    topic_slug: str,
    target_level: str,
    mongo_db: Any
) -> List[str]:
    """Retrieves all concept_ids from MongoDB concept_graphs for levels preceding target_level."""
    if target_level not in LEVEL_ORDER or mongo_db is None:
        return []
    
    target_idx = LEVEL_ORDER.index(target_level)
    previous_levels = LEVEL_ORDER[:target_idx]
    
    ids = set()
    collection = mongo_db.get_collection("concept_graphs")
    
    for lvl in previous_levels:
        doc_id = f"{topic_slug}_{lvl}"
        graph = await collection.find_one({"_id": doc_id})
        if graph:
            nodes = graph.get("nodes") or graph.get("concepts") or []
            for node in nodes:
                cid = node.get("concept_id") or node.get("id")
                if cid:
                    ids.add(str(cid).strip().lower())
    return sorted(list(ids))

async def get_or_create_level_graph(
    topic: str,
    level: str = "beginner",
    mongo_db: Any = None,
    previous_concept_ids: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Retrieves level-specific canonical concept graph from MongoDB or generates a new one dynamically via LLM."""
    norm = normalize_topic(topic)
    level_clean = level.strip().lower()
    doc_id = f"{norm['slug']}_{level_clean}"
    
    collection = mongo_db.get_collection("concept_graphs") if mongo_db is not None else None
    
    if collection is not None:
        existing = await collection.find_one({"_id": doc_id})
        if existing:
            nodes = existing.get("nodes") or existing.get("concepts") or []
            if len(nodes) >= 6:
                return existing
            logger.info(f"[get_or_create_level_graph] Cached graph '{doc_id}' has only {len(nodes)} nodes (< 6). Invalidating cache to generate full 8-node graph...")

        # Fuzzy search fallback for existing level graph
        all_graphs = await collection.find({"level": level_clean}).to_list(100)
        for g in all_graphs:
            aliases = g.get("aliases", []) + [g.get("topic", ""), g.get("canonical_topic", "")]
            match = process.extractOne(norm["canonical_topic"], aliases, scorer=fuzz.token_sort_ratio)
            if match and match[1] >= 85:
                nodes = g.get("nodes") or g.get("concepts") or []
                if len(nodes) >= 6:
                    logger.info(f"Fuzzy matched level graph '{norm['canonical_topic']}' to '{g['_id']}'")
                    return g

    # Build clean excluded concept IDs set
    excluded_ids_set = set()
    if previous_concept_ids:
        for p_item in previous_concept_ids:
            if isinstance(p_item, str):
                excluded_ids_set.add(p_item.strip().lower())
            elif isinstance(p_item, dict) and (p_item.get("concept_id") or p_item.get("id")):
                excluded_ids_set.add(str(p_item.get("concept_id") or p_item.get("id")).strip().lower())

    # Also automatically include concept IDs from MongoDB concept_graphs for preceding levels
    if mongo_db is not None and level_clean in LEVEL_ORDER and level_clean != "beginner":
        auto_prev_ids = await get_previous_concept_ids(norm["slug"], level_clean, mongo_db)
        excluded_ids_set.update(auto_prev_ids)

    # Generate canonical level graph dynamically via LLM (up to 2 attempts on empty/invalid result)
    prev_concepts_str = json.dumps(sorted(list(excluded_ids_set))) if excluded_ids_set else "[]"

    last_gen_error: Optional[Exception] = None
    for attempt in range(1, 3):
        try:
            res = await call_llm_gateway(
                CURRICULUM_GRAPH_PROMPT,
                {
                    "topic": norm["canonical_topic"],
                    "target_level": level_clean,
                    "previous_level_concepts": prev_concepts_str
                },
                ConceptGraphSchema
            )

            raw_nodes = res.nodes or res.concepts or []
            concepts_data = []
            generated_ids = set()

            for c in raw_nodes:
                c_dict = c.model_dump() if hasattr(c, "model_dump") else dict(c)
                cid = c_dict.get("concept_id") or c_dict.get("id")
                if not cid or not str(cid).strip():
                    logger.warning(f"[get_or_create_level_graph] Skipping node missing concept_id: {c_dict}")
                    continue
                cid = str(cid).strip().lower().replace(" ", "_")

                # Soft-skip duplicate or previously-excluded concepts instead of hard crashing
                if cid in excluded_ids_set:
                    logger.warning(f"[get_or_create_level_graph] Skipping excluded concept_id '{cid}' regenerated by LLM")
                    continue
                if cid in generated_ids:
                    logger.warning(f"[get_or_create_level_graph] Skipping duplicate concept_id '{cid}' in same graph")
                    continue

                generated_ids.add(cid)
                c_dict["id"] = cid
                c_dict["concept_id"] = cid
                c_dict["node_type"] = c_dict.get("node_type") if c_dict.get("node_type") in ["learning", "practice"] else "learning"
                c_dict["pattern"] = c_dict.get("pattern") or "none"
                valid_types = ["code", "conceptual", "mcq", "short_answer", "math"]
                c_type = c_dict.get("concept_type")
                if c_type not in valid_types:
                    c_type = "conceptual" if c_dict["node_type"] == "learning" else random.choice(["code", "mcq", "conceptual", "short_answer"])
                c_dict["concept_type"] = c_type
                concepts_data.append(c_dict)

            if not concepts_data:
                raise ValueError(f"LLM returned 0 valid concept nodes for level '{level_clean}' (attempt {attempt})")

            # Validate node correctness & schema integrity
            validate_concept_nodes(concepts_data, level=level_clean)

            # Topological cycle validation & dependency pruning with previous prerequisite IDs preserved
            validate_topological_sort(concepts_data, valid_prereq_ids=excluded_ids_set)

            doc = {
                "_id": doc_id,
                "topic": norm["slug"],
                "canonical_topic": norm["canonical_topic"],
                "level": level_clean,
                "aliases": res.aliases or [],
                "concepts": concepts_data,
                "nodes": concepts_data
            }

            if collection is not None:
                await collection.update_one({"_id": doc_id}, {"$set": doc}, upsert=True)

            return doc

        except Exception as gen_err:
            last_gen_error = gen_err
            logger.warning(f"[get_or_create_level_graph] Attempt {attempt} failed: {gen_err}. {'Retrying...' if attempt < 2 else 'Giving up.'}")

    raise RuntimeError(
        f"Failed to generate concept graph for topic='{norm['canonical_topic']}' level='{level_clean}' after 2 attempts. Last error: {last_gen_error}"
    )


async def get_or_create_concept_graph(
    topic: str,
    mongo_db: Any,
    target_level: str = "expert",
    knowledge_profile: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Generates concept graphs for all levels up to target_level concurrently and
    merges them into a single flat concept list for use by the diagnostic engine.
    This ensures the knowledge_profile contains concepts from all levels (beginner
    through expert), not just the beginner graph.
    """
    import asyncio
    norm = normalize_topic(topic)

    # Determine which levels to generate based on target_level
    target_idx = LEVEL_ORDER.index(target_level) if target_level in LEVEL_ORDER else len(LEVEL_ORDER) - 1
    levels_to_generate = LEVEL_ORDER[: target_idx + 1]  # e.g. ["beginner", "intermediate", "advanced", "expert"]

    # Generate all level graphs concurrently for speed
    tasks = [
        get_or_create_level_graph(topic=norm["canonical_topic"], level=lvl, mongo_db=mongo_db)
        for lvl in levels_to_generate
    ]
    level_graphs = await asyncio.gather(*tasks, return_exceptions=True)

    # Merge all nodes from all levels into a single flat list (de-duplicated by concept_id)
    seen_ids: set = set()
    all_concepts: list = []
    all_aliases: list = []

    for result in level_graphs:
        if isinstance(result, Exception):
            logger.warning(f"[get_or_create_concept_graph] Level graph generation failed: {result}")
            continue
        nodes = result.get("nodes") or result.get("concepts") or []
        for node in nodes:
            cid = node.get("concept_id") or node.get("id")
            if cid and cid not in seen_ids:
                seen_ids.add(cid)
                all_concepts.append(node)
        for alias in result.get("aliases", []):
            if alias not in all_aliases:
                all_aliases.append(alias)

    if not all_concepts:
        raise RuntimeError(
            f"Failed to generate any concept graph levels for topic='{norm['canonical_topic']}'"
        )

    logger.info(
        f"[get_or_create_concept_graph] Merged {len(all_concepts)} concepts across "
        f"{len(levels_to_generate)} levels for topic='{norm['canonical_topic']}'"
    )

    return {
        "topic": norm["slug"],
        "canonical_topic": norm["canonical_topic"],
        "target_level": target_level,
        "aliases": all_aliases,
        "concepts": all_concepts,
        "nodes": all_concepts,
    }


async def get_diagnostic_checklist(
    topic: str,
    mongo_db: Any,
    user_id: Optional[Any] = None
) -> List[Dict[str, Any]]:
    """
    Returns exactly 15 diagnostic concept entries [{concept_id, level}, ...] for a topic.

    Priority order (fastest to slowest):
    1. User's existing learning_path nodes in MongoDB  → 0 LLM calls, <5ms
    2. Cached diagnostic_checklists collection         → 0 LLM calls, <10ms
    3. Single lightweight LLM call                    → 1 LLM call, ~5-15s

    The diagnostic engine only needs concept_id + level — it does NOT need the
    full ConceptGraphSchema (dependencies, patterns, complexity, node_types).
    Keeping this separate from concept_graphs avoids triggering the expensive
    4-level full-graph generation during diagnostic startup.
    """
    norm = normalize_topic(topic)
    MAX_CONCEPTS = 15
    LEVELS_ORDER = ["beginner", "intermediate", "advanced", "expert"]
    PER_LEVEL_LIMIT = 4

    # ----------------------------------------------------------------
    # TIER 1: User already has a learning path for this topic
    # Pull learning-type nodes from their existing path (0 LLM calls)
    # ----------------------------------------------------------------
    if mongo_db is not None and user_id is not None:
        try:
            lp_collection = mongo_db.get_collection("learning_paths")
            lp_doc = await lp_collection.find_one({
                "user_id": str(user_id),
                "topic": {"$in": [norm["slug"], norm["canonical_topic"], topic.strip().lower()]}
            })
            if lp_doc:
                lp_nodes = lp_doc.get("nodes", [])
                by_level: Dict[str, List] = {lvl: [] for lvl in LEVELS_ORDER}
                for node in lp_nodes:
                    if node.get("node_type", "learning") != "learning":
                        continue
                    cid = node.get("concept_id") or node.get("id")
                    lvl = node.get("level", "beginner")
                    if cid and lvl in by_level:
                        by_level[lvl].append({"concept_id": cid, "level": lvl})

                checklist: List[Dict] = []
                for lvl in LEVELS_ORDER:
                    checklist.extend(by_level[lvl][:PER_LEVEL_LIMIT])
                    if len(checklist) >= MAX_CONCEPTS:
                        break
                checklist = checklist[:MAX_CONCEPTS]

                if checklist:
                    logger.info(
                        f"[get_diagnostic_checklist] TIER-1 HIT: {len(checklist)} concepts "
                        f"from existing learning_path for topic='{norm['slug']}'"
                    )
                    return checklist
        except Exception as e:
            logger.warning(f"[get_diagnostic_checklist] Tier-1 learning_path lookup failed: {e}")

    # ----------------------------------------------------------------
    # TIER 2: Cached diagnostic_checklists collection in MongoDB
    # ----------------------------------------------------------------
    cache_id = f"diag_checklist:{norm['slug']}"
    if mongo_db is not None:
        try:
            cl_collection = mongo_db.get_collection("diagnostic_checklists")
            cached = await cl_collection.find_one({"_id": cache_id})
            if cached and cached.get("concepts"):
                logger.info(
                    f"[get_diagnostic_checklist] TIER-2 HIT: cached checklist for topic='{norm['slug']}'"
                )
                return cached["concepts"]
        except Exception as e:
            logger.warning(f"[get_diagnostic_checklist] Tier-2 cache lookup failed: {e}")

    # ----------------------------------------------------------------
    # TIER 3: Single lightweight LLM call — returns only concept_ids
    # (no dependencies, no patterns, no full graph — just names + levels)
    # ----------------------------------------------------------------
    logger.info(f"[get_diagnostic_checklist] TIER-3: LLM call for topic='{norm['canonical_topic']}'")
    try:
        res = await call_llm_gateway(
            DIAGNOSTIC_CHECKLIST_PROMPT,
            {"topic": norm["canonical_topic"]},
            DiagnosticChecklistSchema
        )

        # Distribute proportionally across levels, cap at MAX_CONCEPTS
        by_level = {lvl: [] for lvl in LEVELS_ORDER}
        for item in res.concepts or []:
            cid = getattr(item, "concept_id", None)
            lvl = getattr(item, "level", "beginner")
            if cid and lvl in by_level:
                by_level[lvl].append({"concept_id": cid.strip().lower(), "level": lvl})

        checklist = []
        for lvl in LEVELS_ORDER:
            checklist.extend(by_level[lvl][:PER_LEVEL_LIMIT])
            if len(checklist) >= MAX_CONCEPTS:
                break
        checklist = checklist[:MAX_CONCEPTS]

        if not checklist:
            raise ValueError("LLM returned empty concept checklist")

        # Cache result permanently in MongoDB
        if mongo_db is not None:
            try:
                cl_collection = mongo_db.get_collection("diagnostic_checklists")
                await cl_collection.update_one(
                    {"_id": cache_id},
                    {"$set": {"_id": cache_id, "topic": norm["slug"], "concepts": checklist}},
                    upsert=True
                )
                logger.info(f"[get_diagnostic_checklist] Cached checklist for '{norm['slug']}'")
            except Exception as save_err:
                logger.warning(f"[get_diagnostic_checklist] Failed to cache checklist: {save_err}")

        return checklist

    except Exception as e:
        logger.error(f"[get_diagnostic_checklist] LLM call failed: {e}. Returning fallback checklist.")
        slug = norm["slug"]
        return [
            {"concept_id": f"{slug}_fundamentals", "level": "beginner"},
            {"concept_id": f"{slug}_data_structures", "level": "beginner"},
            {"concept_id": f"{slug}_core_mechanics", "level": "beginner"},
            {"concept_id": f"{slug}_common_patterns", "level": "intermediate"},
            {"concept_id": f"{slug}_implementation_techniques", "level": "intermediate"},
            {"concept_id": f"{slug}_advanced_optimization", "level": "advanced"},
            {"concept_id": f"{slug}_system_design", "level": "expert"},
        ]



async def generate_personalized_learning_path(
    user_id: uuid.UUID,
    topic: str,
    target_level: str,
    knowledge_profile: Dict[str, Dict[str, Any]],
    diagnostic_session_id: Optional[uuid.UUID],
    mongo_db: Any,
    db: AsyncSession
) -> Dict[str, Any]:
    """Generates personalized learning path starting with the BEGINNER level graph from the beginning."""
    norm = normalize_topic(topic)
    initial_level = "beginner"
    user_id_str = str(user_id)

    level_graph = await get_or_create_level_graph(
        topic=norm["canonical_topic"],
        level=initial_level,
        mongo_db=mongo_db
    )
    raw_nodes = level_graph.get("nodes") or level_graph.get("concepts", [])

    # 1. Reset user_topics completed nodes for this user and topic so freshly generated path starts clean
    if mongo_db is not None:
        ut_collection = mongo_db.get_collection("user_topics")
        await ut_collection.update_one(
            {"user_id": user_id_str, "topic_slug": norm["slug"]},
            {
                "$set": {
                    "canonical_topic": norm["canonical_topic"],
                    "raw_input_topic": norm["user_topic"],
                    "current_level": initial_level,
                    "completed_nodes": [],
                    "completed_node_ids": [],
                    "updated_at": datetime.now(timezone.utc).isoformat()
                },
                "$setOnInsert": {
                    "aliases": [norm["slug"], norm["canonical_topic"].lower()],
                    "status": "active",
                    "created_at": datetime.now(timezone.utc).isoformat()
                }
            },
            upsert=True
        )

    # Reset Postgres UserTopic if exists
    if db is not None:
        stmt_ut = select(UserTopic).where(
            UserTopic.user_id == user_id,
            UserTopic.topic_slug == norm["slug"]
        )
        res_ut = await db.execute(stmt_ut)
        ut_row = res_ut.scalar_one_or_none()
        if ut_row:
            ut_row.completed_nodes = []
            ut_row.completed_node_ids = []
            ut_row.current_level = initial_level

    # 2. Clean up any existing learning path documents for this user and topic so stale documents don't persist
    if mongo_db is not None:
        lp_collection = mongo_db.get_collection("learning_paths")
        await lp_collection.delete_many({
            "user_id": user_id_str,
            "topic": {"$in": [norm["slug"], norm["canonical_topic"].lower()]}
        })

    nodes = []
    skipped_concepts = []

    for c in raw_nodes:
        c_id = c.get("concept_id") or c.get("id")
        if not c_id or not isinstance(c_id, str) or not c_id.strip():
            raise ValueError(f"Concept node missing valid concept_id: {c}")
        c_id = c_id.strip().lower()

        prof = knowledge_profile.get(c_id, {"confidence": 0.0, "status": "unknown"})
        conf = prof.get("confidence", 0.0)

        node_type = c.get("node_type") if c.get("node_type") in ["learning", "practice"] else "learning"

        # Calculate node status purely based on user's diagnostic profile and prerequisites
        # Never inherit 'mastered' or previous completion from shared graph templates
        has_prereqs = bool(c.get("depends_on"))
        if conf >= 0.8:
            status = "known"
            action = "skip_teach_verify_only"
            est = 0
            skipped_concepts.append(c_id)
        elif conf >= 0.4:
            status = "partial"
            action = "refresher"
            est = 10
        else:
            status = "locked" if has_prereqs else "learning"
            action = "full_teach"
            est = c.get("estimated_minutes") or (15 if node_type == "learning" else 30)

        target_q_count = 0 if node_type == "learning" else (c.get("target_question_count") or 5)

        c_type = c.get("concept_type")
        if c_type not in ["code", "conceptual", "mcq", "short_answer", "math"]:
            c_type = "conceptual" if node_type == "learning" else random.choice(["code", "mcq", "conceptual", "short_answer"])

        node_obj = {
            "concept_id": c_id,
            "level": initial_level,
            "node_type": node_type,
            "concept_type": c_type,
            "pattern": c.get("pattern", "none"),
            "status": status,
            "action": action,
            "estimated_minutes": est,
            "target_question_count": target_q_count,
            "depends_on": c.get("depends_on", [])
        }
        nodes.append(node_obj)

        # Seed/Reset Postgres concept_mastery table so user starts fresh with no pre-mastered starting nodes
        stmt = select(ConceptMastery).where(
            ConceptMastery.user_id == user_id,
            ConceptMastery.topic == norm["slug"],
            ConceptMastery.concept_id == c_id
        )
        res = await db.execute(stmt)
        existing_mastery = res.scalar_one_or_none()

        if not existing_mastery:
            mastery_row = ConceptMastery(
                user_id=user_id,
                topic=norm["slug"],
                concept_id=c_id,
                mastery=conf,
                attempts=0,
                failures=0,
                status=status
            )
            db.add(mastery_row)
        else:
            existing_mastery.status = status
            existing_mastery.mastery = conf
            existing_mastery.attempts = 0
            existing_mastery.failures = 0

    await db.commit()

    completed_ids = {n["concept_id"] for n in nodes if n["status"] == "known"}
    available_nodes = [n["concept_id"] for n in get_available_nodes(nodes, completed_ids)]
    first_active = available_nodes[0] if available_nodes else (nodes[0]["concept_id"] if nodes else None)

    lp_id = f"lp_{uuid.uuid4().hex[:8]}"
    learning_path_doc = {
        "_id": lp_id,
        "user_id": user_id_str,
        "topic": norm["slug"],
        "canonical_topic": norm["canonical_topic"],
        "current_level": initial_level,
        "target_level": target_level,
        "generated_from_diagnostic_session": str(diagnostic_session_id) if diagnostic_session_id else None,
        "completed_levels": [],
        "nodes": nodes,
        "available_nodes": available_nodes,
        "current_node": first_active
    }

    if mongo_db is not None:
        lp_collection = mongo_db.get_collection("learning_paths")
        await lp_collection.insert_one(learning_path_doc)

    return learning_path_doc

async def generate_next_level_graph(
    learning_path_id: str,
    user_id: uuid.UUID,
    mongo_db: Any,
    db: AsyncSession
) -> Dict[str, Any]:
    """Generates next level graph and appends nodes to the existing learning path document."""
    lp_collection = mongo_db.get_collection("learning_paths")
    lp_doc = await lp_collection.find_one({"_id": learning_path_id, "user_id": str(user_id)})

    if not lp_doc:
        raise ValueError(f"Learning path '{learning_path_id}' not found for user '{user_id}'")

    current_level = lp_doc.get("current_level", "beginner")
    curr_idx = LEVEL_ORDER.index(current_level) if current_level in LEVEL_ORDER else 0
    if curr_idx >= len(LEVEL_ORDER) - 1:
        logger.info(f"Learning path '{learning_path_id}' is already at maximum level '{current_level}'")
        return lp_doc

    next_level = LEVEL_ORDER[curr_idx + 1]
    topic_slug = lp_doc.get("topic")
    canonical_topic = lp_doc.get("canonical_topic", topic_slug.replace("-", " ").title())

    existing_nodes = lp_doc.get("nodes", [])
    existing_ids = {
        n["concept_id"]
        for n in existing_nodes
        if n.get("concept_id")
    }

    next_graph = await get_or_create_level_graph(
        topic=canonical_topic,
        level=next_level,
        mongo_db=mongo_db,
        previous_concept_ids=list(existing_ids)
    )

    new_raw_nodes = next_graph.get("nodes") or next_graph.get("concepts", [])
    new_node_objs = []

    for c in new_raw_nodes:
        c_id = c.get("concept_id") or c.get("id")
        if not c_id or not isinstance(c_id, str) or not c_id.strip():
            raise ValueError(f"Next level concept node missing valid concept_id: {c}")
        c_id = c_id.strip().lower()

        if c_id in existing_ids:
            continue

        node_type = c.get("node_type") if c.get("node_type") in ["learning", "practice"] else "learning"
        target_q_count = 0 if node_type == "learning" else (c.get("target_question_count") or 5)

        has_prereqs = bool(c.get("depends_on"))
        completed_set = {n["concept_id"] for n in existing_nodes if n.get("status") in ["known", "mastered"]}
        deps_met = set(c.get("depends_on", [])).issubset(completed_set)

        status = "learning" if (not has_prereqs or deps_met) else "locked"

        c_type = c.get("concept_type")
        if c_type not in ["code", "conceptual", "mcq", "short_answer", "math"]:
            c_type = "conceptual" if node_type == "learning" else random.choice(["code", "mcq", "conceptual", "short_answer"])

        node_obj = {
            "concept_id": c_id,
            "level": next_level,
            "node_type": node_type,
            "concept_type": c_type,
            "pattern": c.get("pattern", "none"),
            "status": status,
            "action": "full_teach",
            "estimated_minutes": c.get("estimated_minutes", 15),
            "target_question_count": target_q_count,
            "depends_on": c.get("depends_on", [])
        }
        new_node_objs.append(node_obj)

        # Seed Postgres concept_mastery
        stmt = select(ConceptMastery).where(
            ConceptMastery.user_id == user_id,
            ConceptMastery.topic == topic_slug,
            ConceptMastery.concept_id == c_id
        )
        m_res = await db.execute(stmt)
        if not m_res.scalar_one_or_none():
            db.add(ConceptMastery(
                user_id=user_id,
                topic=topic_slug,
                concept_id=c_id,
                mastery=0.0,
                attempts=0,
                failures=0,
                status=status
            ))

    await db.commit()

    all_nodes = existing_nodes + new_node_objs
    completed_levels = lp_doc.get("completed_levels", [])
    if current_level not in completed_levels:
        completed_levels.append(current_level)

    completed_ids = {n["concept_id"] for n in all_nodes if n.get("status") in ["known", "mastered"]}
    avail = [n["concept_id"] for n in get_available_nodes(all_nodes, completed_ids)]
    first_active = avail[0] if avail else (all_nodes[0]["concept_id"] if all_nodes else None)

    updated_fields = {
        "current_level": next_level,
        "completed_levels": completed_levels,
        "nodes": all_nodes,
        "available_nodes": avail,
        "current_node": first_active
    }

    await lp_collection.update_one({"_id": learning_path_id}, {"$set": updated_fields})
    lp_doc.update(updated_fields)

    # Sync current_level to user_topics for this user and topic
    ut_collection = mongo_db.get_collection("user_topics")
    await ut_collection.update_one(
        {"user_id": str(user_id), "topic_slug": topic_slug},
        {"$set": {"current_level": next_level, "updated_at": datetime.now(timezone.utc).isoformat()}}
    )

    return lp_doc

async def sync_learning_path_progress(
    user_id: Any,
    topic_slug: str,
    mongo_db: Any,
    db: AsyncSession,
    learning_path_id: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Synchronizes concept mastery from PostgreSQL to MongoDB learning path,
    unlocks downstream nodes whose prerequisites are satisfied, recalculates
    available_nodes and current_node, and automatically triggers next-level
    generation if the current level's concepts are fully mastered.
    """
    if mongo_db is None or db is None:
        return None

    user_id_str = str(user_id)
    lp_collection = mongo_db.get_collection("learning_paths")
    lp_doc = None

    if learning_path_id:
        lp_doc = await lp_collection.find_one({"_id": learning_path_id, "user_id": user_id_str})

    if not lp_doc:
        lp_doc = await lp_collection.find_one({"user_id": user_id_str, "topic": topic_slug})

    if not lp_doc:
        lp_doc = await lp_collection.find_one({"user_id": user_id_str, "topic": topic_slug.lower()})

    if not lp_doc:
        return None

    nodes = lp_doc.get("nodes", [])
    if not nodes:
        return lp_doc

    uid_uuid = uuid.UUID(user_id_str) if isinstance(user_id, str) else user_id
    stmt = select(ConceptMastery).where(
        ConceptMastery.user_id == uid_uuid,
        ConceptMastery.topic == topic_slug
    )
    res = await db.execute(stmt)
    mastery_records = res.scalars().all()
    mastery_map = {r.concept_id: r.status for r in mastery_records}

    # Fetch user's completed nodes specifically from user_topics
    ut_collection = mongo_db.get_collection("user_topics")
    ut_doc = await ut_collection.find_one({"user_id": user_id_str, "topic_slug": topic_slug})
    user_completed_ids = set((ut_doc.get("completed_node_ids") if ut_doc else []) or [])

    completed_ids = set(user_completed_ids)
    for n in nodes:
        cid = n.get("concept_id") or n.get("id")
        if not cid:
            continue

        pg_status = mastery_map.get(cid)
        target_q_count = n.get("target_question_count", 3)
        correct_count = n.get("correct_count", 0)

        if cid in user_completed_ids:
            n["status"] = "mastered"
            completed_ids.add(cid)
        elif pg_status in ["known", "mastered"]:
            n["status"] = pg_status
            completed_ids.add(cid)
        elif target_q_count > 0 and correct_count >= target_q_count:
            n["status"] = "mastered"
            completed_ids.add(cid)
            await record_user_completed_node(user_id, topic_slug, cid, mongo_db, db)

    for n in nodes:
        cid = n.get("concept_id") or n.get("id")
        if n.get("status") not in ["known", "mastered"]:
            deps = set(n.get("depends_on", []))
            if not deps or deps.issubset(completed_ids):
                if n.get("status") == "locked":
                    n["status"] = "learning"

    avail_nodes = [n["concept_id"] for n in get_available_nodes(nodes, completed_ids)]
    current_node = avail_nodes[0] if avail_nodes else (nodes[-1].get("concept_id") if nodes else None)

    current_level = lp_doc.get("current_level", "beginner")
    target_level = lp_doc.get("target_level", "expert")

    level_concept_ids = [
        n["concept_id"] for n in nodes
        if (n.get("level") or "beginner").lower() == (current_level or "beginner").lower() and n.get("concept_id")
    ]
    level_completed = bool(level_concept_ids and all(cid in completed_ids for cid in level_concept_ids))

    if level_completed and current_level != target_level:
        curr_idx = LEVEL_ORDER.index(current_level) if current_level in LEVEL_ORDER else 0
        target_idx = LEVEL_ORDER.index(target_level) if target_level in LEVEL_ORDER else len(LEVEL_ORDER) - 1

        if curr_idx < target_idx and curr_idx < len(LEVEL_ORDER) - 1:
            logger.info(f"Auto-progressing level for user '{user_id_str}' topic '{topic_slug}' from '{current_level}'")
            lp_id = lp_doc["_id"]
            updated_doc = await generate_next_level_graph(
                learning_path_id=lp_id,
                user_id=uid_uuid,
                mongo_db=mongo_db,
                db=db
            )
            return updated_doc

    updated_fields = {
        "nodes": nodes,
        "available_nodes": avail_nodes,
        "current_node": current_node
    }

    await lp_collection.update_one({"_id": lp_doc["_id"]}, {"$set": updated_fields})
    lp_doc.update(updated_fields)

    # Sync current_level to user_topics for this user and topic
    ut_collection = mongo_db.get_collection("user_topics")
    await ut_collection.update_one(
        {"user_id": user_id_str, "topic_slug": topic_slug},
        {"$set": {"current_level": current_level, "updated_at": datetime.now(timezone.utc).isoformat()}}
    )

    return lp_doc

def shuffle_checkpoint_options(content: Dict[str, Any]) -> Dict[str, Any]:
    """Cleans up embedded options from question stem and shuffles MCQ options array so correct_option is randomly positioned."""
    import re
    if not isinstance(content, dict):
        return content

    checkpoint = content.get("checkpoint")
    if isinstance(checkpoint, dict):
        raw_q_text = str(checkpoint.get("question") or "").strip()
        opts = checkpoint.get("options")
        correct = checkpoint.get("correct_option")

        # 1. Check for options embedded in the question stem
        extracted = []
        clean_lines = []
        opt_line_pattern = re.compile(r"^\s*(?:([A-D])|([1-4]))[\)\.\:\-]\s*(.+)$", re.IGNORECASE)

        for line in raw_q_text.split("\n"):
            m = opt_line_pattern.match(line)
            if m:
                let = (m.group(1) or "").upper()
                dig = m.group(2) or ""
                key = let if let else ("ABCD"[int(dig) - 1] if dig and 1 <= int(dig) <= 4 else "")
                extracted.append((key, m.group(3).strip()))
            else:
                clean_lines.append(line)

        if len(extracted) < 2:
            inline_parts = re.split(r"\s+(?=(?:[A-D]|[1-4])[\)\.\:\-]\s+)", raw_q_text)
            if len(inline_parts) >= 3:
                temp_extracted = []
                for part in inline_parts[1:]:
                    m = opt_line_pattern.match(part.strip())
                    if m:
                        let = (m.group(1) or "").upper()
                        dig = m.group(2) or ""
                        key = let if let else ("ABCD"[int(dig) - 1] if dig and 1 <= int(dig) <= 4 else "")
                        temp_extracted.append((key, m.group(3).strip()))
                if len(temp_extracted) >= 2:
                    extracted = temp_extracted
                    clean_lines = [inline_parts[0].strip()]

        raw_opts_list = list(opts or []) if isinstance(opts, list) else []
        are_raw_opts_letters = not raw_opts_list or all(
            len(str(o).strip()) <= 2 and (str(o).strip().upper() in ["A", "B", "C", "D"] or str(o).strip() in ["1", "2", "3", "4"])
            for o in raw_opts_list
        )

        if extracted and (are_raw_opts_letters or len(raw_opts_list) < len(extracted)):
            checkpoint["question"] = "\n".join(clean_lines).strip()
            opts = [text for _, text in extracted]
            corr_str = str(correct or "").strip()
            for key, text in extracted:
                if corr_str.upper() == key or corr_str == text or corr_str.upper().startswith(key):
                    correct = text
                    checkpoint["correct_option"] = text
                    break
            checkpoint["options"] = opts

        # 2. Normalize letter/index-based correct_option to exact string text before shuffling
        if isinstance(opts, list) and len(opts) > 1 and correct:
            correct_str = str(correct).strip()

            if correct_str.upper() in ["A", "B", "C", "D"] and len(opts) == 4:
                idx = ["A", "B", "C", "D"].index(correct_str.upper())
                checkpoint["correct_option"] = opts[idx]
            elif len(correct_str) >= 2 and correct_str[:2].upper() in ["A)", "B)", "C)", "D)"]:
                idx = ["A)", "B)", "C)", "D)"].index(correct_str[:2].upper())
                if idx < len(opts):
                    checkpoint["correct_option"] = opts[idx]
            elif correct_str.lower().startswith("option ") and len(correct_str) >= 8 and correct_str[7].upper() in ["A", "B", "C", "D"]:
                idx = ["A", "B", "C", "D"].index(correct_str[7].upper())
                if idx < len(opts):
                    checkpoint["correct_option"] = opts[idx]

            # Create a copy and shuffle options randomly
            shuffled_opts = list(opts)
            random.shuffle(shuffled_opts)
            checkpoint["options"] = shuffled_opts

    return content


async def generate_learning_content(
    topic: str,
    concept_id: str,
    level: str = "beginner",
    pattern: Optional[str] = None,
    prerequisites: Optional[List[str]] = None,
    language: str = "python",
    explanation_language: str = "english",
    mongo_db: Any = None,
    force_regenerate: bool = False
) -> Dict[str, Any]:
    """Generates structured learning lesson content with Redis & MongoDB caching for instant response times."""
    norm = normalize_topic(topic)
    pattern_clean = pattern or "none"
    
    # Validate and sanitize language parameter to prevent payload bugs (e.g. topic passed as language)
    valid_langs = {"python", "javascript", "js", "typescript", "cpp", "c++", "java", "go", "rust", "csharp", "c#"}
    lang_input = (language or "python").lower().strip()
    lang_clean = lang_input if lang_input in valid_langs else "python"
    exp_lang_clean = (explanation_language or "english").lower().strip()
    
    cache_key = f"lesson_v4:{norm['slug']}:{concept_id}:{level}:{pattern_clean}:{lang_clean}:{exp_lang_clean}"

    # 1. Check Redis Cache (< 5ms response time) unless force_regenerate is True
    from app.db.redis_client import get_redis_client
    redis_c = get_redis_client()
    if force_regenerate:
        if redis_c:
            try:
                await redis_c.delete(cache_key)
            except Exception:
                pass
        if mongo_db is not None:
            try:
                lessons_col = mongo_db.get_collection("lessons")
                await lessons_col.delete_one({"_id": cache_key})
            except Exception:
                pass
    else:
        if redis_c:
            try:
                cached_data = await redis_c.get(cache_key)
                if cached_data:
                    logger.info(f"[generate_learning_content] Redis Cache HIT for '{cache_key}'")
                    return shuffle_checkpoint_options(json.loads(cached_data))
            except Exception as re_err:
                logger.warning(f"Redis lookup failed for key '{cache_key}': {re_err}")

        # 2. Check MongoDB persistence cache
        if mongo_db is not None:
            try:
                lessons_col = mongo_db.get_collection("lessons")
                existing_doc = await lessons_col.find_one({"_id": cache_key})
                if existing_doc and "content" in existing_doc:
                    logger.info(f"[generate_learning_content] MongoDB Cache HIT for '{cache_key}'")
                    content = existing_doc["content"]
                    if redis_c:
                        try:
                            await redis_c.set(cache_key, json.dumps(content), ex=86400)
                        except Exception:
                            pass
                    return shuffle_checkpoint_options(content)
            except Exception as mg_err:
                logger.warning(f"MongoDB lookup failed for key '{cache_key}': {mg_err}")

    # 3. Cache MISS -> Call LLM Gateway
    prereq_str = str(prerequisites) if prerequisites else "None"
    res = await call_llm_gateway(
        LEARNING_CONTENT_PROMPT,
        {
            "topic": norm["canonical_topic"],
            "concept_id": concept_id,
            "level": level,
            "pattern": pattern_clean,
            "prerequisites": prereq_str,
            "language": lang_clean,
            "explanation_language": exp_lang_clean
        },
        LessonContentSchema
    )

    content = res.model_dump() if hasattr(res, "model_dump") else dict(res)

    # 4. Save to Redis & MongoDB cache for instant future loads
    if redis_c:
        try:
            await redis_c.set(cache_key, json.dumps(content), ex=86400)
        except Exception as save_err:
            logger.warning(f"Failed to cache lesson in Redis: {save_err}")

    if mongo_db is not None:
        try:
            lessons_col = mongo_db.get_collection("lessons")
            await lessons_col.update_one(
                {"_id": cache_key},
                {"$set": {"_id": cache_key, "topic": norm["slug"], "concept_id": concept_id, "language": lang_clean, "content": content}},
                upsert=True
            )
        except Exception as save_mg_err:
            logger.warning(f"Failed to cache lesson in MongoDB: {save_mg_err}")

    return shuffle_checkpoint_options(content)


async def generate_lesson_overview(
    topic: str,
    concept_id: str,
    level: str = "beginner",
    pattern: Optional[str] = None,
    prerequisites: Optional[List[str]] = None,
    language: str = "python",
    explanation_language: str = "english",
    mongo_db: Any = None
) -> Dict[str, Any]:
    """Generates Part 1 (Core overview & compilable code) with Redis cache."""
    from app.llm.prompts import LESSON_OVERVIEW_PROMPT
    from app.llm.schemas import LessonOverviewSchema

    norm = normalize_topic(topic)
    pattern_clean = pattern or "none"
    valid_langs = {"python", "javascript", "js", "typescript", "cpp", "c++", "java", "go", "rust", "csharp", "c#"}
    lang_input = (language or "python").lower().strip()
    lang_clean = lang_input if lang_input in valid_langs else "python"
    exp_lang_clean = (explanation_language or "english").lower().strip()

    cache_key = f"lesson_ov:{norm['slug']}:{concept_id}:{level}:{pattern_clean}:{lang_clean}:{exp_lang_clean}"

    from app.db.redis_client import get_redis_client
    redis_c = get_redis_client()
    if redis_c:
        try:
            cached_data = await redis_c.get(cache_key)
            if cached_data:
                return json.loads(cached_data)
        except Exception:
            pass

    prereq_str = str(prerequisites) if prerequisites else "None"
    res = await call_llm_gateway(
        LESSON_OVERVIEW_PROMPT,
        {
            "topic": norm["canonical_topic"],
            "concept_id": concept_id,
            "level": level,
            "pattern": pattern_clean,
            "prerequisites": prereq_str,
            "language": lang_clean,
            "explanation_language": exp_lang_clean
        },
        LessonOverviewSchema
    )
    content = res.model_dump() if hasattr(res, "model_dump") else dict(res)

    if redis_c:
        try:
            await redis_c.set(cache_key, json.dumps(content), ex=86400)
        except Exception:
            pass
    return content


async def generate_lesson_deep_dive(
    topic: str,
    concept_id: str,
    level: str = "beginner",
    pattern: Optional[str] = None,
    prerequisites: Optional[List[str]] = None,
    language: str = "python",
    explanation_language: str = "english",
    mongo_db: Any = None
) -> Dict[str, Any]:
    """Generates Part 2 (Walkthrough, complexity, edge cases & checkpoint) with Redis cache."""
    from app.llm.prompts import LESSON_DEEP_DIVE_PROMPT
    from app.llm.schemas import LessonDeepDiveSchema

    norm = normalize_topic(topic)
    pattern_clean = pattern or "none"
    valid_langs = {"python", "javascript", "js", "typescript", "cpp", "c++", "java", "go", "rust", "csharp", "c#"}
    lang_input = (language or "python").lower().strip()
    lang_clean = lang_input if lang_input in valid_langs else "python"
    exp_lang_clean = (explanation_language or "english").lower().strip()

    cache_key = f"lesson_dd:{norm['slug']}:{concept_id}:{level}:{pattern_clean}:{lang_clean}:{exp_lang_clean}"

    from app.db.redis_client import get_redis_client
    redis_c = get_redis_client()
    if redis_c:
        try:
            cached_data = await redis_c.get(cache_key)
            if cached_data:
                return shuffle_checkpoint_options(json.loads(cached_data))
        except Exception:
            pass

    prereq_str = str(prerequisites) if prerequisites else "None"
    res = await call_llm_gateway(
        LESSON_DEEP_DIVE_PROMPT,
        {
            "topic": norm["canonical_topic"],
            "concept_id": concept_id,
            "level": level,
            "pattern": pattern_clean,
            "prerequisites": prereq_str,
            "language": lang_clean,
            "explanation_language": exp_lang_clean
        },
        LessonDeepDiveSchema
    )
    content = res.model_dump() if hasattr(res, "model_dump") else dict(res)

    if redis_c:
        try:
            await redis_c.set(cache_key, json.dumps(content), ex=86400)
        except Exception:
            pass
    return shuffle_checkpoint_options(content)