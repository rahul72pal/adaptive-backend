import logging
import uuid
from typing import Optional, List, Any
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete

from app.db.postgres import get_db
from app.db.mongo import get_mongo_db
from app.core.dependencies import get_current_user
from app.db.models import User, UserTopic, ConceptMastery, DiagnosticSession
from app.engine.curriculum_engine import get_or_create_user_topic, normalize_topic, slugify_topic
from app.catalog.models import (
    TopicMetadata,
    CategoryMetadata,
    CategoryCatalogResponse,
    TopicCatalogResponse,
    TopicResolutionResult,
    TopicResolutionResponse,
)
from app.services.topic_catalog_service import get_catalog_service, TopicCatalogService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/topics", tags=["Topic Management & Catalog"])


class SelectTopicSchema(BaseModel):
    topic: str = Field(..., description="Stable topic_id (e.g. t_08_binary_search) or legacy topic name")
    category_id: Optional[str] = Field(None, description="Optional category_id for disambiguation (e.g. cat_08)")


class SetTopicModeSchema(BaseModel):
    topic: str = Field(..., description="Topic ID or topic name")
    mode_choice: str = Field(..., description="'diagnostic' or 'beginner'")
    diagnostic_session_id: Optional[str] = None
    category_id: Optional[str] = None


# =============================================================================
# 1. CANONICAL CATALOG ENDPOINTS
# =============================================================================

@router.get("/categories", response_model=CategoryCatalogResponse)
async def list_categories(
    catalog_service: TopicCatalogService = Depends(get_catalog_service)
):
    """
    Returns all 60 canonical DSA categories with topic counts, default taxonomic kinds,
    and assessment configurations.
    """
    cats = catalog_service.get_all_categories()
    return CategoryCatalogResponse(
        total_categories=len(cats),
        categories=cats
    )


@router.get("/catalog", response_model=TopicCatalogResponse)
async def list_catalog_topics(
    category_id: Optional[str] = Query(None, description="Filter by category_id (e.g. cat_05) or full category name"),
    kind: Optional[str] = Query(None, description="Filter by kind: data_structure, algorithm, technique, concrete_problem, math_theory, systems_practice, proof_skill"),
    assessment_mode: Optional[str] = Query(None, description="Filter by mode: code, conceptual_only, mixed"),
    level: Optional[str] = Query(None, description="Filter by level: beginner, intermediate, advanced, expert"),
    search: Optional[str] = Query(None, description="Keyword search across topic names, patterns, and categories"),
    page: int = Query(1, ge=1, description="Page number (1-indexed)"),
    page_size: int = Query(50, ge=1, le=2500, description="Page size limit"),
    catalog_service: TopicCatalogService = Depends(get_catalog_service)
):
    """
    Lists canonical topics from the metadata-driven catalog with multi-axis filtering,
    case-insensitive keyword search, and pagination.
    """
    topics, total = catalog_service.get_topics(
        category_id=category_id,
        kind=kind,
        assessment_mode=assessment_mode,
        level=level,
        search=search,
        page=page,
        page_size=page_size
    )
    return TopicCatalogResponse(
        total_topics=total,
        page=page,
        page_size=page_size,
        topics=topics
    )


@router.get("/catalog/{topic_id}", response_model=TopicMetadata)
async def get_topic_by_id(
    topic_id: str,
    catalog_service: TopicCatalogService = Depends(get_catalog_service)
):
    """
    Retrieves full metadata for a topic by its stable, collision-free identifier (e.g. 't_08_binary_search').
    """
    topic = catalog_service.get_topic_by_id(topic_id)
    if not topic:
        # Fallback: check if it resolves as an alias or name
        resolution = catalog_service.resolve_topic(topic_id)
        if resolution.is_resolved and resolution.topic:
            topic = resolution.topic
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Topic '{topic_id}' was not found in the canonical catalog."
            )
    return topic


@router.get("/catalog/{topic_id}/metadata")
async def get_topic_metadata_for_generation(
    topic_id: str,
    catalog_service: TopicCatalogService = Depends(get_catalog_service)
):
    """
    Returns sanitized metadata needed specifically for curriculum graph generation,
    diagnostic question synthesis, and adaptive lesson instruction.
    """
    metadata = catalog_service.get_topic_metadata_for_generation(topic_id)
    if not metadata:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Topic metadata for '{topic_id}' could not be resolved."
        )
    return metadata


@router.get("/resolve", response_model=TopicResolutionResponse)
async def resolve_topic(
    topic: str = Query(..., description="Raw topic string, slug, or legacy name to resolve"),
    category_id: Optional[str] = Query(None, description="Optional category_id to resolve ambiguities"),
    catalog_service: TopicCatalogService = Depends(get_catalog_service)
):
    """
    Resolves legacy topic names, aliases, or slugs to their canonical TopicMetadata.
    Ambiguous topics return candidates and an actionable error rather than selecting wrongly.
    """
    result = catalog_service.resolve_topic(topic, preferred_category_id=category_id)
    return TopicResolutionResponse(
        status="resolved" if result.is_resolved else ("ambiguous" if result.is_ambiguous else "unresolved"),
        result=result
    )


@router.post("/catalog/sync", status_code=status.HTTP_200_OK)
async def sync_catalog_to_database_endpoint(
    db: AsyncSession = Depends(get_db),
    mongo_db: Any = Depends(get_mongo_db),
    catalog_service: TopicCatalogService = Depends(get_catalog_service)
):
    """
    Admin/Internal endpoint to idempotently sync the canonical topic catalog into
    PostgreSQL and MongoDB if not already populated. Never alters learner data.
    """
    res = await catalog_service.sync_to_database(db=db, mongo_db=mongo_db)
    return {"status": "success", "synced": res}


# =============================================================================
# 2. USER TOPIC SELECTION & MANAGEMENT (BACKWARD COMPATIBLE)
# =============================================================================

@router.post("/select", status_code=status.HTTP_200_OK)
async def select_or_create_topic(
    payload: SelectTopicSchema,
    current_user: User = Depends(get_current_user),
    mongo_db: Any = Depends(get_mongo_db),
    catalog_service: TopicCatalogService = Depends(get_catalog_service)
):
    """
    Selects or creates a topic linked to the authenticated user in MongoDB 'user_topics'.
    
    Supports:
    1. Stable topic_id (e.g. 't_08_binary_search')
    2. Canonical display name (e.g. 'Binary Search')
    3. Legacy alias or slug (e.g. 'binary-search', 'Array')
    4. Custom non-DSA topics outside the standard catalog
    
    If a legacy name is ambiguous across multiple categories, produces a clear
    validation error requiring disambiguation.
    """
    raw_topic = payload.topic.strip()
    if not raw_topic:
        raise HTTPException(status_code=400, detail="Topic name cannot be empty.")

    # 1. Resolve via Canonical TopicCatalogService
    resolution = catalog_service.resolve_topic(
        raw_topic,
        preferred_category_id=payload.category_id
    )

    if resolution.is_ambiguous:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=resolution.message
        )

    topic_metadata: Optional[TopicMetadata] = None
    if resolution.is_resolved and resolution.topic:
        topic_metadata = resolution.topic
        effective_canonical_name = topic_metadata.name
        effective_slug = topic_metadata.topic_id
    else:
        # Non-catalog / custom user topic (e.g. external library)
        norm = normalize_topic(raw_topic)
        effective_canonical_name = norm["canonical_topic"]
        effective_slug = norm["slug"]

    try:
        topic_doc = await get_or_create_user_topic(
            user_id=current_user.id,
            raw_topic=effective_canonical_name,
            mongo_db=mongo_db
        )

        # Check if user has already completed diagnostic assessment for this topic
        lp_collection = mongo_db.get_collection("learning_paths")
        slug_to_check = topic_doc.get("topic_slug", effective_slug)
        lp_doc = await lp_collection.find_one({
            "user_id": str(current_user.id),
            "topic": {"$in": [slug_to_check, effective_slug, raw_topic.lower()]}
        })

        is_completed = bool(lp_doc and (lp_doc.get("generated_from_diagnostic_session") or lp_doc.get("completed_levels")))

        topic_slug = topic_doc.get("topic_slug") or slugify_topic(effective_canonical_name)
        canonical_topic = topic_metadata.name if topic_metadata else (topic_doc.get("canonical_topic") or effective_canonical_name)

        # Update mongo user_topic with metadata fields if available
        if topic_metadata and mongo_db is not None:
            try:
                ut_col = mongo_db.get_collection("user_topics")
                await ut_col.update_one(
                    {"_id": topic_doc["_id"]},
                    {
                        "$set": {
                            "topic_id": topic_metadata.topic_id,
                            "category_id": topic_metadata.category_id,
                            "category_name": topic_metadata.category_name,
                        },
                        "$addToSet": {
                            "aliases": topic_metadata.topic_id
                        }
                    }
                )
            except Exception as update_err:
                logger.warning(f"Failed to persist topic_id on user_topic: {update_err}")

        response_data = {
            # Legacy fields (preserved 100% backward compatible)
            "topic_slug": topic_slug,
            "canonical_topic": canonical_topic,
            "raw_input_topic": topic_doc.get("raw_input_topic", raw_topic),
            "aliases": topic_doc.get("aliases", []),
            "is_completed": is_completed,
            "mode_choice": topic_doc.get("mode_choice"),
            "diagnostic_session_id": topic_doc.get("diagnostic_session_id"),
            "current_level": topic_doc.get("current_level", "beginner"),
            "completed_nodes": topic_doc.get("completed_nodes", []),
            "completed_node_ids": topic_doc.get("completed_node_ids", []),
            # New metadata fields
            "topic_id": topic_metadata.topic_id if topic_metadata else topic_slug,
            "category_id": topic_metadata.category_id if topic_metadata else "cat_custom",
            "category_name": topic_metadata.category_name if topic_metadata else "Custom Topic",
            "kind": topic_metadata.kind if topic_metadata else "technique",
            "scope": topic_metadata.scope if topic_metadata else "topic",
            "assessment_mode": topic_metadata.assessment_mode if topic_metadata else "mixed",
            "supported_levels": topic_metadata.supported_levels if topic_metadata else ["beginner", "intermediate", "advanced", "expert"],
            "default_target_level": topic_metadata.default_target_level if topic_metadata else "expert",
            "pattern_tags": topic_metadata.pattern_tags if topic_metadata else [],
            "target_node_count": topic_metadata.generation_config.target_node_count if topic_metadata else 8,
            "metadata_version": topic_metadata.metadata_version if topic_metadata else 1,
        }
        return response_data

    except Exception as e:
        logger.error(f"[select_or_create_topic] Error for topic '{raw_topic}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to select topic: {str(e)}")


@router.post("/mode", status_code=status.HTTP_200_OK)
async def set_topic_mode_choice(
    payload: SetTopicModeSchema,
    current_user: User = Depends(get_current_user),
    mongo_db: Any = Depends(get_mongo_db),
    catalog_service: TopicCatalogService = Depends(get_catalog_service)
):
    """
    Stores the user's path mode choice ('diagnostic' or 'beginner') and optional diagnostic_session_id
    for the specified topic in MongoDB 'user_topics'.
    """
    raw_topic = payload.topic.strip()
    if not raw_topic:
        raise HTTPException(status_code=400, detail="Topic name cannot be empty.")
    if payload.mode_choice not in ["diagnostic", "beginner"]:
        raise HTTPException(status_code=400, detail="mode_choice must be 'diagnostic' or 'beginner'.")

    # Resolve topic name if ambiguous
    resolution = catalog_service.resolve_topic(raw_topic, preferred_category_id=payload.category_id)
    if resolution.is_ambiguous and mongo_db is not None:
        try:
            # Check user_topics for this user to see if they already have one of the candidates selected
            ut_collection = mongo_db.get_collection("user_topics")
            user_id_str = str(current_user.id)
            cursor = ut_collection.find({"user_id": user_id_str})
            user_docs = await cursor.to_list(100) if hasattr(cursor, "to_list") else []

            matching = None
            for u_doc in user_docs:
                u_topic_id = u_doc.get("topic_id")
                u_topic_slug = u_doc.get("topic_slug")
                u_aliases = u_doc.get("aliases", [])
                u_cat_id = u_doc.get("category_id")

                for cand in resolution.matching_candidates:
                    if (
                        cand.topic_id == u_topic_id
                        or cand.topic_id == u_topic_slug
                        or cand.topic_id in u_aliases
                        or (u_cat_id and cand.category_id == u_cat_id)
                    ):
                        matching = cand
                        break
                if matching:
                    break

            if matching:
                resolution = TopicResolutionResult(
                    is_resolved=True,
                    topic=matching,
                    message=f"Disambiguated via user's selected topic '{matching.topic_id}'."
                )
        except Exception as e:
            logger.warning(f"Failed to auto-disambiguate ambiguous topic from user_topics: {e}")

    if resolution.is_ambiguous:
        raise HTTPException(status_code=400, detail=resolution.message)

    effective_topic = resolution.topic.name if (resolution.is_resolved and resolution.topic) else raw_topic

    try:
        topic_doc = await get_or_create_user_topic(
            user_id=current_user.id,
            raw_topic=effective_topic,
            mongo_db=mongo_db
        )

        ut_collection = mongo_db.get_collection("user_topics")
        user_id_str = str(current_user.id)
        topic_slug = topic_doc["topic_slug"]

        update_fields = {
            "mode_choice": payload.mode_choice,
            "updated_at": datetime.now(timezone.utc).isoformat()
        }
        if payload.diagnostic_session_id:
            update_fields["diagnostic_session_id"] = payload.diagnostic_session_id

        await ut_collection.update_one(
            {"user_id": user_id_str, "topic_slug": topic_slug},
            {"$set": update_fields}
        )

        return {
            "status": "success",
            "topic_slug": topic_slug,
            "mode_choice": payload.mode_choice,
            "diagnostic_session_id": payload.diagnostic_session_id or topic_doc.get("diagnostic_session_id")
        }
    except Exception as e:
        logger.error(f"[set_topic_mode_choice] Error setting mode for topic '{raw_topic}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to set topic mode choice: {str(e)}")


@router.get("/user")
async def get_user_topics(
    current_user: User = Depends(get_current_user),
    mongo_db: Any = Depends(get_mongo_db),
    catalog_service: TopicCatalogService = Depends(get_catalog_service)
):
    """
    Retrieves all topics linked to the authenticated user from MongoDB 'user_topics' collection,
    enriched with canonical catalog metadata.
    """
    try:
        ut_collection = mongo_db.get_collection("user_topics")
        user_id_str = str(current_user.id)
        
        # Query active user topics
        docs = await ut_collection.find({
            "user_id": user_id_str,
            "status": {"$ne": "deleted"}
        }).to_list(100)

        # Check completion status in learning_paths
        lp_collection = mongo_db.get_collection("learning_paths")
        lp_docs = await lp_collection.find({"user_id": user_id_str}).to_list(100)
        completed_slugs = {
            doc.get("topic") for doc in lp_docs
            if doc.get("generated_from_diagnostic_session") or doc.get("completed_levels")
        }

        # Format user topics list enriched with catalog metadata
        user_topics = []
        for d in docs:
            t_slug = d.get("topic_slug", "")
            canon_name = d.get("canonical_topic", t_slug.title())
            
            # Resolve against catalog
            res = catalog_service.resolve_topic(canon_name)
            meta = res.topic if (res.is_resolved and res.topic) else None

            user_topics.append({
                "id": meta.topic_id if meta else t_slug,
                "topic_id": meta.topic_id if meta else t_slug,
                "topic_slug": t_slug,
                "canonical_topic": canon_name,
                "name": canon_name,
                "raw_input_topic": d.get("raw_input_topic", t_slug),
                "aliases": d.get("aliases", []),
                "category_id": meta.category_id if meta else "cat_custom",
                "category_name": meta.category_name if meta else "Custom Curriculum",
                "kind": meta.kind if meta else "technique",
                "assessment_mode": meta.assessment_mode if meta else "mixed",
                "is_completed": t_slug in completed_slugs,
                "mode_choice": d.get("mode_choice"),
                "diagnostic_session_id": d.get("diagnostic_session_id"),
                "current_level": d.get("current_level", "beginner"),
                "completed_nodes": d.get("completed_nodes", []),
                "completed_node_ids": d.get("completed_node_ids", []),
                "icon": "🎓",
                "description": meta.description if meta else f"Custom AI-generated learning curriculum for {canon_name}."
            })

        # Default fallback if user has no saved topics in DB yet
        if not user_topics:
            default_doc = await get_or_create_user_topic(current_user.id, "Array", mongo_db)
            res_arr = catalog_service.resolve_topic("Array")
            meta_arr = res_arr.topic if (res_arr.is_resolved and res_arr.topic) else None

            user_topics.append({
                "id": meta_arr.topic_id if meta_arr else default_doc["topic_slug"],
                "topic_id": meta_arr.topic_id if meta_arr else default_doc["topic_slug"],
                "topic_slug": default_doc["topic_slug"],
                "canonical_topic": default_doc["canonical_topic"],
                "name": default_doc["canonical_topic"],
                "raw_input_topic": "Array",
                "aliases": default_doc.get("aliases", ["array"]),
                "category_id": meta_arr.category_id if meta_arr else "cat_05",
                "category_name": meta_arr.category_name if meta_arr else "05. Arrays",
                "kind": meta_arr.kind if meta_arr else "data_structure",
                "assessment_mode": meta_arr.assessment_mode if meta_arr else "mixed",
                "is_completed": False,
                "mode_choice": default_doc.get("mode_choice"),
                "diagnostic_session_id": default_doc.get("diagnostic_session_id"),
                "current_level": default_doc.get("current_level", "beginner"),
                "completed_nodes": default_doc.get("completed_nodes", []),
                "completed_node_ids": default_doc.get("completed_node_ids", []),
                "icon": "🎓",
                "description": meta_arr.description if meta_arr else "Custom AI-generated learning curriculum for Array."
            })

        return {"topics": user_topics}
    except Exception as e:
        logger.error(f"[get_user_topics] Error for user {current_user.id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to fetch user topics: {str(e)}")


@router.delete("/user/{topic_slug}")
async def delete_user_topic(
    topic_slug: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    mongo_db: Any = Depends(get_mongo_db),
    catalog_service: TopicCatalogService = Depends(get_catalog_service)
):
    """
    Completely removes a topic link and all associated user details for the authenticated user:
    - Removes user_topics from MongoDB and PostgreSQL
    - Removes user's personalized learning path from MongoDB
    - Removes user's concept mastery records from PostgreSQL
    - Removes user's diagnostic sessions for this topic from PostgreSQL
    """
    try:
        user_id_str = str(current_user.id)
        norm = normalize_topic(topic_slug)
        clean_slug = norm["slug"]
        slug_candidates = list(set([
            clean_slug,
            topic_slug.strip().lower(),
            norm["canonical_topic"].lower()
        ]))

        # Also add topic_id and aliases if it matches catalog
        res = catalog_service.resolve_topic(topic_slug)
        if res.is_resolved and res.topic:
            slug_candidates.append(res.topic.topic_id)
            slug_candidates.extend(res.topic.aliases)

        slug_candidates = list(set(slug_candidates))

        # 1. Delete user_topics and learning_paths from MongoDB
        if mongo_db is not None:
            ut_collection = mongo_db.get_collection("user_topics")
            await ut_collection.delete_many({
                "user_id": user_id_str,
                "$or": [
                    {"topic_slug": {"$in": slug_candidates}},
                    {"aliases": {"$in": slug_candidates}}
                ]
            })

            lp_collection = mongo_db.get_collection("learning_paths")
            await lp_collection.delete_many({
                "user_id": user_id_str,
                "topic": {"$in": slug_candidates}
            })

        # 2. Delete user_topics, concept_mastery, and diagnostic_sessions from PostgreSQL
        if db is not None:
            stmt_ut = delete(UserTopic).where(
                UserTopic.user_id == current_user.id,
                UserTopic.topic_slug.in_(slug_candidates)
            )
            await db.execute(stmt_ut)

            stmt_cm = delete(ConceptMastery).where(
                ConceptMastery.user_id == current_user.id,
                ConceptMastery.topic.in_(slug_candidates)
            )
            await db.execute(stmt_cm)

            stmt_ds = delete(DiagnosticSession).where(
                DiagnosticSession.user_id == current_user.id,
                DiagnosticSession.topic.in_(slug_candidates)
            )
            await db.execute(stmt_ds)

            await db.commit()

        return {
            "status": "success",
            "message": f"Topic '{clean_slug}' and all associated user topic details, learning paths, and progress have been deleted."
        }
    except Exception as e:
        logger.error(f"[delete_user_topic] Error deleting topic '{topic_slug}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to delete topic: {str(e)}")
