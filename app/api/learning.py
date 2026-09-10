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
from app.engine.curriculum_engine import generate_personalized_learning_path
from app.graphs.reteach_graph import select_strategy_node

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/learning", tags=["Learning Path & Curriculum Engine"])

class GeneratePathSchema(BaseModel):
    topic: str
    target_level: str = "expert"  # beginner, intermediate, advanced, expert
    diagnostic_session_id: Optional[str] = None

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
    """Generates personalized learning path based on user's diagnostic knowledge profile."""
    logger.info(
        f"[generate_path] Step 1: Received request for topic='{payload.topic}', "
        f"target_level='{payload.target_level}', diag_session_id='{payload.diagnostic_session_id}' "
        f"from user_id={current_user.id}"
    )

    knowledge_profile = {}
    diag_uuid = None

    if payload.diagnostic_session_id:
        try:
            logger.info(f"[generate_path] Step 2: Querying diagnostic session ID '{payload.diagnostic_session_id}'")
            diag_uuid = uuid.UUID(payload.diagnostic_session_id)
            stmt = select(DiagnosticSession).where(DiagnosticSession.id == diag_uuid)
            res = await db.execute(stmt)
            session_rec = res.scalar_one_or_none()
            if session_rec:
                knowledge_profile = session_rec.knowledge_profile or {}
                logger.info(
                    f"[generate_path] Step 2b: Found diagnostic session. "
                    f"Loaded knowledge profile ({len(knowledge_profile)} concepts)"
                )
            else:
                logger.warning(f"[generate_path] Step 2b: Diagnostic session ID '{payload.diagnostic_session_id}' not found in DB.")
        except ValueError as err:
            logger.warning(f"[generate_path] Step 2b: Invalid diagnostic_session_id format: {err}")

    try:
        logger.info(f"[generate_path] Step 3: Invoking generate_personalized_learning_path engine for topic='{payload.topic}'")
        path_doc = await generate_personalized_learning_path(
            user_id=current_user.id,
            topic=payload.topic,
            target_level=payload.target_level,
            knowledge_profile=knowledge_profile,
            diagnostic_session_id=diag_uuid,
            mongo_db=mongo_db,
            db=db
        )
        path_id = path_doc.get("_id") if isinstance(path_doc, dict) else getattr(path_doc, "id", None)
        logger.info(f"[generate_path] Step 4: Learning path created successfully! Path ID='{path_id}'")
        return path_doc
    except Exception as e:
        logger.error(f"[generate_path] Error generating learning path for topic='{payload.topic}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate learning path: {str(e)}")

@router.get("/{user_id}/{topic}")
async def get_learning_path(
    user_id: str,
    topic: str,
    current_user: User = Depends(get_current_user),
    mongo_db: Any = Depends(get_mongo_db)
):
    """Retrieves current personalized learning path for user and topic."""
    lp_collection = mongo_db.get_collection("learning_paths")
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
