import logging
import uuid
from typing import Optional, List, Any
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete

from app.db.postgres import get_db
from app.db.mongo import get_mongo_db
from app.core.dependencies import get_current_user
from app.db.models import User, UserTopic, ConceptMastery, DiagnosticSession
from app.engine.curriculum_engine import get_or_create_user_topic, normalize_topic, slugify_topic

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/topics", tags=["Topic Management & User Topics"])

from datetime import datetime, timezone

class SelectTopicSchema(BaseModel):
    topic: str

class SetTopicModeSchema(BaseModel):
    topic: str
    mode_choice: str  # "diagnostic" | "beginner"
    diagnostic_session_id: Optional[str] = None

@router.post("/select", status_code=status.HTTP_200_OK)
async def select_or_create_topic(
    payload: SelectTopicSchema,
    current_user: User = Depends(get_current_user),
    mongo_db: Any = Depends(get_mongo_db)
):
    """
    Selects or creates a topic linked to the authenticated user in MongoDB 'user_topics'.
    Performs alias and fuzzy matching so variations (e.g. 'js in Array', 'Arrays') link to canonical 'Array'.
    """
    raw_topic = payload.topic.strip()
    if not raw_topic:
        raise HTTPException(status_code=400, detail="Topic name cannot be empty.")

    try:
        topic_doc = await get_or_create_user_topic(
            user_id=current_user.id,
            raw_topic=raw_topic,
            mongo_db=mongo_db
        )

        # Check if user has already completed diagnostic assessment for this topic
        lp_collection = mongo_db.get_collection("learning_paths")
        lp_doc = await lp_collection.find_one({
            "user_id": str(current_user.id),
            "topic": topic_doc["topic_slug"]
        })

        is_completed = bool(lp_doc and (lp_doc.get("generated_from_diagnostic_session") or lp_doc.get("completed_levels")))

        topic_slug = topic_doc.get("topic_slug") or slugify_topic(raw_topic)
        canonical_topic = topic_doc.get("canonical_topic") or raw_topic.replace("-", " ").replace("_", " ").title()

        return {
            "topic_slug": topic_slug,
            "canonical_topic": canonical_topic,
            "raw_input_topic": topic_doc.get("raw_input_topic", raw_topic),
            "aliases": topic_doc.get("aliases", []),
            "is_completed": is_completed,
            "mode_choice": topic_doc.get("mode_choice"),
            "diagnostic_session_id": topic_doc.get("diagnostic_session_id"),
            "current_level": topic_doc.get("current_level", "beginner"),
            "completed_nodes": topic_doc.get("completed_nodes", []),
            "completed_node_ids": topic_doc.get("completed_node_ids", [])
        }
    except Exception as e:
        logger.error(f"[select_or_create_topic] Error for topic '{raw_topic}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to select topic: {str(e)}")


@router.post("/mode", status_code=status.HTTP_200_OK)
async def set_topic_mode_choice(
    payload: SetTopicModeSchema,
    current_user: User = Depends(get_current_user),
    mongo_db: Any = Depends(get_mongo_db)
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

    try:
        topic_doc = await get_or_create_user_topic(
            user_id=current_user.id,
            raw_topic=raw_topic,
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
    mongo_db: Any = Depends(get_mongo_db)
):
    """Retrieves all topics linked to the authenticated user from MongoDB 'user_topics' collection."""
    try:
        ut_collection = mongo_db.get_collection("user_topics")
        user_id_str = str(current_user.id)
        
        # Query active user topics (include documents without explicit status field or where status != "deleted")
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

        # Format user topics list
        user_topics = []
        for d in docs:
            t_slug = d.get("topic_slug", "")
            user_topics.append({
                "id": t_slug,
                "topic_slug": t_slug,
                "canonical_topic": d.get("canonical_topic", t_slug.title()),
                "name": d.get("canonical_topic", t_slug.title()),
                "raw_input_topic": d.get("raw_input_topic", t_slug),
                "aliases": d.get("aliases", []),
                "is_completed": t_slug in completed_slugs,
                "mode_choice": d.get("mode_choice"),
                "diagnostic_session_id": d.get("diagnostic_session_id"),
                "current_level": d.get("current_level", "beginner"),
                "completed_nodes": d.get("completed_nodes", []),
                "completed_node_ids": d.get("completed_node_ids", []),
                "icon": "🎓",
                "description": f"Custom AI-generated learning curriculum for {d.get('canonical_topic', t_slug)}."
            })

        # Default fallback if user has no saved topics in DB yet
        if not user_topics:
            default_doc = await get_or_create_user_topic(current_user.id, "Array", mongo_db)
            user_topics.append({
                "id": default_doc["topic_slug"],
                "topic_slug": default_doc["topic_slug"],
                "canonical_topic": default_doc["canonical_topic"],
                "name": default_doc["canonical_topic"],
                "raw_input_topic": "Array",
                "aliases": default_doc.get("aliases", ["array"]),
                "is_completed": False,
                "mode_choice": default_doc.get("mode_choice"),
                "diagnostic_session_id": default_doc.get("diagnostic_session_id"),
                "current_level": default_doc.get("current_level", "beginner"),
                "completed_nodes": default_doc.get("completed_nodes", []),
                "completed_node_ids": default_doc.get("completed_node_ids", []),
                "icon": "🎓",
                "description": "Custom AI-generated learning curriculum for Array."
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
    mongo_db: Any = Depends(get_mongo_db)
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
