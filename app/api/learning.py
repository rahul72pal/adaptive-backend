import logging
import uuid
from typing import Optional, Any, List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel

from app.db.postgres import get_db
from app.db.mongo import get_mongo_db
from app.core.dependencies import get_current_user
from app.db.models import User, DiagnosticSession
from app.engine.curriculum_engine import (
    generate_personalized_learning_path,
    generate_next_level_graph,
    generate_learning_content,
    sync_learning_path_progress,
    slugify_topic,
    get_available_nodes
)
from app.engine.mastery_engine import check_level_completion
from app.graphs.reteach_graph import select_strategy_node

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/learning", tags=["Learning Path & Curriculum Engine"])

class GeneratePathSchema(BaseModel):
    topic: str
    target_level: str = "expert"  # beginner, intermediate, advanced, expert
    diagnostic_session_id: Optional[str] = None

class CompleteLevelSchema(BaseModel):
    learning_path_id: str

class LessonRequestSchema(BaseModel):
    topic: str
    concept_id: str
    level: str = "beginner"
    pattern: Optional[str] = "none"
    prerequisites: List[str] = []

class ReteachRequestSchema(BaseModel):
    topic: str
    concept_id: str
    strategies_tried: List[str] = []
    consecutive_failures: int = 1

@router.post("/generate-path", status_code=status.HTTP_201_CREATED)
async def generate_path(
    payload: GeneratePathSchema,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    mongo_db: Any = Depends(get_mongo_db)
):
    """Generates personalized learning path starting with the beginner level graph."""
    logger.info(
        f"[generate_path] Request topic='{payload.topic}', target_level='{payload.target_level}', "
        f"diag_session='{payload.diagnostic_session_id}' from user_id={current_user.id}"
    )

    knowledge_profile = {}
    diag_uuid = None

    if payload.diagnostic_session_id:
        try:
            diag_uuid = uuid.UUID(payload.diagnostic_session_id)
            stmt = select(DiagnosticSession).where(DiagnosticSession.id == diag_uuid)
            res = await db.execute(stmt)
            session_rec = res.scalar_one_or_none()
            if session_rec:
                knowledge_profile = session_rec.knowledge_profile or {}
        except ValueError as err:
            logger.warning(f"[generate_path] Invalid diagnostic_session_id format: {err}")

    try:
        path_doc = await generate_personalized_learning_path(
            user_id=current_user.id,
            topic=payload.topic,
            target_level=payload.target_level,
            knowledge_profile=knowledge_profile,
            diagnostic_session_id=diag_uuid,
            mongo_db=mongo_db,
            db=db
        )
        return path_doc
    except Exception as e:
        logger.error(f"[generate_path] Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate learning path: {str(e)}")

@router.post("/complete-level")
async def complete_level_and_progress(
    payload: CompleteLevelSchema,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    mongo_db: Any = Depends(get_mongo_db)
):
    """Triggers generation of the next level graph when the user completes the current level."""
    try:
        updated_doc = await generate_next_level_graph(
            learning_path_id=payload.learning_path_id,
            user_id=current_user.id,
            mongo_db=mongo_db,
            db=db
        )
        return updated_doc
    except ValueError as val_err:
        raise HTTPException(status_code=404, detail=str(val_err))
    except Exception as e:
        logger.error(f"[complete_level_and_progress] Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to progress level: {str(e)}")

@router.post("/lesson")
async def get_lesson_content(
    payload: LessonRequestSchema,
    current_user: User = Depends(get_current_user),
    mongo_db: Any = Depends(get_mongo_db)
):
    """Generates structured lesson content for a learning node using LEARNING_CONTENT_PROMPT."""
    try:
        content = await generate_learning_content(
            topic=payload.topic,
            concept_id=payload.concept_id,
            level=payload.level,
            pattern=payload.pattern,
            prerequisites=payload.prerequisites,
            mongo_db=mongo_db
        )
        return content
    except Exception as e:
        logger.error(f"[get_lesson_content] Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate lesson content: {str(e)}")

@router.get("/{user_id}/{topic}")
async def get_learning_path(
    user_id: str,
    topic: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    mongo_db: Any = Depends(get_mongo_db)
):
    """Retrieves current personalized learning path for user and topic."""
    slug = slugify_topic(topic)
    
    path_doc = await sync_learning_path_progress(
        user_id=user_id,
        topic_slug=slug,
        mongo_db=mongo_db,
        db=db
    )

    if not path_doc:
        lp_collection = mongo_db.get_collection("learning_paths")
        path_doc = await lp_collection.find_one({
            "user_id": user_id,
            "topic": slug
        })
        if not path_doc:
            path_doc = await lp_collection.find_one({
                "user_id": user_id,
                "topic": topic.lower()
            })

    if not path_doc:
        raise HTTPException(status_code=404, detail="Learning path not found for given user and topic.")

    return path_doc

@router.post("/reteach")
async def reteach_concept(
    payload: ReteachRequestSchema,
    current_user: User = Depends(get_current_user)
):
    """Executes stateful LangGraph adaptive re-teaching flow to try a new pedagogical strategy."""
    state = {
        "user_id": str(current_user.id),
        "concept_id": payload.concept_id,
        "topic": payload.topic,
        "strategies_tried": payload.strategies_tried,
        "current_strategy": None,
        "explanation_content": None,
        "consecutive_failures": payload.consecutive_failures,
        "action": "continue"
    }

    result_state = await select_strategy_node(state)
    return result_state
