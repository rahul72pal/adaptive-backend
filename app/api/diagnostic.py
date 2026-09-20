import logging
import uuid
from datetime import datetime, timezone
from typing import Optional, Any
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from pydantic import BaseModel

from app.db.postgres import get_db
from app.db.mongo import get_mongo_db
from app.db.redis_client import get_redis_checkpointer
from app.core.dependencies import get_current_user
from app.db.models import User, DiagnosticSession
from app.engine.curriculum_engine import get_or_create_concept_graph
from app.graphs.diagnostic_graph import build_diagnostic_graph

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/diagnostic", tags=["Diagnostic Chat Engine"])

class StartDiagnosticSchema(BaseModel):
    topic: str

class AnswerDiagnosticSchema(BaseModel):
    session_id: str
    answer: str

class OptOutDiagnosticSchema(BaseModel):
    session_id: str

@router.post("/start", status_code=status.HTTP_201_CREATED)
async def start_diagnostic(
    payload: StartDiagnosticSchema,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    mongo_db: Any = Depends(get_mongo_db),
    checkpointer: Any = Depends(get_redis_checkpointer)
):
    """Initializes a new LangGraph diagnostic session checkpointed in Redis."""
    logger.info(f"[start_diagnostic] Step 1: Initiating start_diagnostic for topic='{payload.topic}' from user_id={current_user.id}")
    try:
        logger.info(f"[start_diagnostic] Step 2: Fetching/generating concept graph for topic='{payload.topic}'")
        concept_graph = await get_or_create_concept_graph(payload.topic, mongo_db)
        
        raw_concepts = concept_graph.get("concepts", []) if isinstance(concept_graph, dict) else getattr(concept_graph, "concepts", [])
        concept_checklist = [
            c["id"] if isinstance(c, dict) else (getattr(c, "id", None) or c.get("id"))
            for c in raw_concepts
        ]
        logger.info(f"[start_diagnostic] Step 3: Extracted {len(concept_checklist)} concepts: {concept_checklist}")
        
        session_id = str(uuid.uuid4())
        thread_config = {"configurable": {"thread_id": session_id}}
        logger.info(f"[start_diagnostic] Step 4: Generated session_id='{session_id}' with thread config")
        
        initial_state = {
            "session_id": session_id,
            "user_id": str(current_user.id),
            "topic": payload.topic,
            "concept_checklist": concept_checklist,
            "knowledge_profile": {c: {"confidence": 0.0, "status": "unassessed"} for c in concept_checklist},
            "current_concept": None,
            "current_question": None,
            "last_user_answer": None,
            "turns_used": 0,
            "is_complete": False
        }

        # Initialize PostgreSQL record
        logger.info(f"[start_diagnostic] Step 5: Creating & committing new DiagnosticSession record in PostgreSQL database")
        new_session = DiagnosticSession(
            id=uuid.UUID(session_id),
            user_id=current_user.id,
            topic=payload.topic,
            knowledge_profile=initial_state["knowledge_profile"],
            turns_used=0
        )
        db.add(new_session)
        await db.commit()
        logger.info(f"[start_diagnostic] Step 5b: Session record committed to DB for session_id='{session_id}'")

        # Run LangGraph through ask_node -> END
        logger.info(f"[start_diagnostic] Step 6: Compiling LangGraph graph_app and invoking initial state (ask_node)")
        graph_app = build_diagnostic_graph(checkpointer)
        state = await graph_app.ainvoke(initial_state, config=thread_config)
        logger.info(f"[start_diagnostic] Step 7: LangGraph execution completed. Generated question='{state.get('current_question')}' for concept='{state.get('current_concept')}'")

        response_payload = {
            "session_id": session_id,
            "turn": state.get("turns_used", 1),
            "current_concept": state.get("current_concept"),
            "current_question": state.get("current_question"),
            "is_complete": False,
            "knowledge_profile": state.get("knowledge_profile", initial_state["knowledge_profile"])
        }
        logger.info(f"[start_diagnostic] Step 8: Returning diagnostic start payload to frontend for session_id='{session_id}'")
        return response_payload
    except Exception as e:
        logger.error(f"[start_diagnostic] Error starting diagnostic for topic='{payload.topic}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to start diagnostic: {str(e)}")

@router.post("/answer")
async def answer_diagnostic(
    payload: AnswerDiagnosticSchema,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    checkpointer: Any = Depends(get_redis_checkpointer)
):
    """Resumes the LangGraph thread, evaluates answer, and returns next question or final profile."""
    thread_config = {"configurable": {"thread_id": payload.session_id}}
    graph_app = build_diagnostic_graph(checkpointer)
    
    current_state = await graph_app.aget_state(thread_config)
    if not current_state.values:
        raise HTTPException(status_code=404, detail="Session expired or not found.")

    # Resume graph execution with the user's answer
    updated_state = await graph_app.ainvoke(
        {"last_user_answer": payload.answer},
        config=thread_config
    )

    is_complete = updated_state.get("is_complete", False)
    turns_used = updated_state.get("turns_used", 0)
    profile = updated_state.get("knowledge_profile", {})

    # Sync progress to PostgreSQL
    update_data = {
        "turns_used": turns_used,
        "knowledge_profile": profile
    }
    if is_complete:
        update_data["completed_at"] = datetime.now(timezone.utc)

    stmt = (
        update(DiagnosticSession)
        .where(DiagnosticSession.id == uuid.UUID(payload.session_id))
        .values(**update_data)
    )
    await db.execute(stmt)
    await db.commit()

    if is_complete:
        return {
            "session_id": payload.session_id,
            "turn": turns_used,
            "is_complete": True,
            "knowledge_profile": profile
        }

    return {
        "session_id": payload.session_id,
        "turn": turns_used,
        "current_concept": updated_state.get("current_concept"),
        "current_question": updated_state.get("current_question"),
        "is_complete": False,
        "knowledge_profile": profile
    }

@router.post("/opt-out")
async def opt_out(
    payload: OptOutDiagnosticSchema,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    checkpointer: Any = Depends(get_redis_checkpointer)
):
    """Skips all 15 diagnostic turns and marks all concepts unknown for beginners."""
    thread_config = {"configurable": {"thread_id": payload.session_id}}
    graph_app = build_diagnostic_graph(checkpointer)
    
    current_state = await graph_app.aget_state(thread_config)
    if not current_state.values:
        raise HTTPException(status_code=404, detail="Session not found.")

    checklist = current_state.values.get("concept_checklist", [])
    zero_profile = {c: {"confidence": 0.0, "status": "unassessed"} for c in checklist}

    await graph_app.aupdate_state(
        thread_config,
        {"knowledge_profile": zero_profile, "is_complete": True}
    )

    stmt = (
        update(DiagnosticSession)
        .where(DiagnosticSession.id == uuid.UUID(payload.session_id))
        .values(
            knowledge_profile=zero_profile,
            completed_at=datetime.now(timezone.utc)
        )
    )
    await db.execute(stmt)
    await db.commit()

    return {
        "session_id": payload.session_id,
        "is_complete": True,
        "knowledge_profile": zero_profile
    }

@router.get("/{session_id}")
async def get_diagnostic_state(
    session_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Retrieves state of a diagnostic session directly from PostgreSQL."""
    try:
        session_uuid = uuid.UUID(session_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid session_id format.")

    stmt = select(DiagnosticSession).where(DiagnosticSession.id == session_uuid)
    res = await db.execute(stmt)
    session_record = res.scalar_one_or_none()

    if not session_record:
        raise HTTPException(status_code=404, detail="Diagnostic session not found.")

    return {
        "session_id": str(session_record.id),
        "user_id": str(session_record.user_id),
        "topic": session_record.topic,
        "turns_used": session_record.turns_used,
        "knowledge_profile": session_record.knowledge_profile,
        "completed_at": session_record.completed_at
    }