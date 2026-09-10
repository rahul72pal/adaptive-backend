import uuid
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.db.models import DiagnosticSession, User
from app.graphs.diagnostic_graph import diagnostic_graph_app
from app.llm.gateway import call_llm_gateway
from app.llm.prompts import DIAGNOSTIC_ASK_PROMPT
from app.llm.schemas import DiagnosticQuestionSchema

async def start_diagnostic_session(
    user_id: uuid.UUID,
    topic: str,
    concept_checklist: list[str],
    db: AsyncSession,
    redis_client: Any
) -> Dict[str, Any]:
    """Starts a new diagnostic chat session using LangGraph and persists session state."""
    session_id = uuid.uuid4()
    
    # Initial knowledge profile: all concepts set to default confidence 0.5 (unknown/unprobed)
    initial_profile = {c: {"confidence": 0.5, "status": "unknown"} for c in concept_checklist}
    first_concept = concept_checklist[0] if concept_checklist else "general"
    initial_profile["_current_concept"] = first_concept
    
    initial_state = {
        "session_id": str(session_id),
        "user_id": str(user_id),
        "topic": topic,
        "concept_checklist": concept_checklist,
        "knowledge_profile": initial_profile,
        "current_concept": first_concept,
        "current_question": None,
        "last_user_answer": None,
        "turns_used": 0,
        "is_complete": False
    }

    # Execute first ask_node in diagnostic graph
    res = await call_llm_gateway(
        DIAGNOSTIC_ASK_PROMPT,
        {
            "concept": initial_state["current_concept"],
            "topic": topic,
            "history": 0
        },
        DiagnosticQuestionSchema
    )
    
    initial_state["current_question"] = res.question
    initial_state["turns_used"] = 1

    # Save session to PostgreSQL
    session_record = DiagnosticSession(
        id=session_id,
        user_id=user_id,
        topic=topic,
        knowledge_profile=initial_profile,
        turns_used=1,
        completed_at=None
    )
    db.add(session_record)
    await db.commit()

    # Save in-flight session to Redis
    cache_key = f"diagnostic_session:{session_id}"
    await redis_client.set(cache_key, str(initial_state), ex=3600)

    clean_profile = {k: v for k, v in initial_profile.items() if not k.startswith("_")}
    return {
        "session_id": str(session_id),
        "topic": topic,
        "current_question": res.question,
        "target_concept": initial_state["current_concept"],
        "knowledge_profile": clean_profile,
        "turns_used": 1,
        "is_complete": False
    }

async def process_diagnostic_answer(
    session_id: uuid.UUID,
    answer: str,
    db: AsyncSession,
    redis_client: Any
) -> Dict[str, Any]:
    """Processes user's answer in diagnostic loop, updates scores, and returns next question or final profile."""
    stmt = select(DiagnosticSession).where(DiagnosticSession.id == session_id)
    result = await db.execute(stmt)
    session_record = result.scalar_one_or_none()
    
    if not session_record:
        raise ValueError(f"Diagnostic session {session_id} not found.")

    raw_profile = dict(session_record.knowledge_profile or {})
    concept_checklist = [k for k in raw_profile.keys() if not k.startswith("_")]
    current_concept = raw_profile.get("_current_concept") or (concept_checklist[0] if concept_checklist else "general")

    # Read state or reconstruct
    state = {
        "session_id": str(session_id),
        "user_id": str(session_record.user_id),
        "topic": session_record.topic,
        "concept_checklist": concept_checklist,
        "knowledge_profile": {k: v for k, v in raw_profile.items() if not k.startswith("_")},
        "current_concept": current_concept,
        "current_question": None,
        "last_user_answer": answer,
        "turns_used": session_record.turns_used,
        "is_complete": session_record.completed_at is not None
    }

    # Step 1: Score previous answer using score_node logic
    from app.graphs.diagnostic_graph import score_node, ask_node, route_diagnostic
    state = await score_node(state)

    # Step 2: Route graph (check max turns or early completion)
    next_step = route_diagnostic(state)

    if next_step == "finalize_node":
        state["is_complete"] = True
        session_record.completed_at = datetime.now(timezone.utc)
    else:
        state = await ask_node(state)

    # Persist updated session and targeted concept to Postgres
    updated_profile = dict(state["knowledge_profile"])
    if state.get("current_concept"):
        updated_profile["_current_concept"] = state["current_concept"]

    session_record.knowledge_profile = updated_profile
    session_record.turns_used = state["turns_used"]
    await db.commit()

    clean_profile = {k: v for k, v in updated_profile.items() if not k.startswith("_")}

    return {
        "session_id": str(session_id),
        "current_question": state.get("current_question"),
        "knowledge_profile": clean_profile,
        "turns_used": state["turns_used"],
        "is_complete": state["is_complete"]
    }

async def opt_out_diagnostic(
    session_id: uuid.UUID,
    db: AsyncSession,
    redis_client: Any
) -> Dict[str, Any]:
    """Complete beginner opt-out option: sets all concept scores to 0.0 unknown and finalizes session instantly."""
    stmt = select(DiagnosticSession).where(DiagnosticSession.id == session_id)
    result = await db.execute(stmt)
    session_record = result.scalar_one_or_none()
    
    if not session_record:
        raise ValueError(f"Diagnostic session {session_id} not found.")

    # Set all concepts to 0.0 unknown
    updated_profile = {
        c: {"confidence": 0.0, "status": "unknown"}
        for c in session_record.knowledge_profile.keys()
    }
    
    session_record.knowledge_profile = updated_profile
    session_record.completed_at = datetime.now(timezone.utc)
    await db.commit()

    return {
        "session_id": str(session_id),
        "knowledge_profile": updated_profile,
        "is_complete": True,
        "message": "Opted out of diagnostic. Set user profile to complete beginner."
    }
