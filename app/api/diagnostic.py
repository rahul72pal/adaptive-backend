import logging
import uuid
from datetime import datetime, timezone
from typing import Optional, Any, Dict, List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from pydantic import BaseModel

from app.db.postgres import get_db
from app.db.mongo import get_mongo_db
from app.db.redis_client import get_redis_checkpointer, get_redis_client
from app.core.dependencies import get_current_user
from app.db.models import User, DiagnosticSession
from app.engine.curriculum_engine import get_diagnostic_checklist
from app.graphs.diagnostic_graph import build_diagnostic_graph

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/diagnostic", tags=["Diagnostic Chat Engine"])

from app.llm.stream_events import (
    MessageStartEvent,
    TextDeltaEvent,
    UserInputEvent,
    UserInputPayload,
    ValidationConstraints,
    MessageEndEvent,
    ErrorEvent,
    ToolStartEvent,
    ToolEndEvent,
    format_sse,
    validate_user_input_submission,
)

class StartDiagnosticSchema(BaseModel):
    topic: str
    session_id: Optional[str] = None

class AnswerDiagnosticSchema(BaseModel):
    session_id: str
    answer: str
    input_id: Optional[str] = None

class OptOutDiagnosticSchema(BaseModel):
    session_id: str


async def persist_diagnostic_state(session_id: str, state_vals: Dict[str, Any], mongo_db: Any = None):
    """Caches diagnostic state in Redis and persists to MongoDB for durability across restarts."""
    try:
        redis_c = get_redis_client()
        if redis_c:
            await redis_c.set(f"diag_state:{session_id}", json.dumps(state_vals), ex=86400 * 30)
    except Exception as re_err:
        logger.warning(f"Failed to cache state in Redis: {re_err}")

    if mongo_db:
        try:
            col = mongo_db.get_collection("diagnostic_states")
            await col.update_one(
                {"_id": session_id},
                {"$set": {"_id": session_id, "state": state_vals, "updated_at": datetime.now(timezone.utc).isoformat()}},
                upsert=True
            )
        except Exception as mg_err:
            logger.warning(f"Failed to persist state in MongoDB: {mg_err}")


async def get_or_restore_diagnostic_state(
    session_id: str,
    db: AsyncSession,
    mongo_db: Any,
    checkpointer: Any
) -> Optional[Dict[str, Any]]:
    """
    Retrieves the LangGraph checkpoint state for a diagnostic session.
    If checkpoint was evicted or lost from memory (e.g. after server restart),
    it dynamically restores and hydrates the state from Redis, MongoDB, and PostgreSQL.
    """
    thread_config = {"configurable": {"thread_id": session_id}}
    graph_app = build_diagnostic_graph(checkpointer)

    current_state = await graph_app.aget_state(thread_config)
    if current_state and current_state.values:
        return current_state.values

    async def _safe_update_graph(cfg, data):
        try:
            await graph_app.aupdate_state(cfg, data, as_node="ask_node")
        except Exception:
            try:
                await graph_app.aupdate_state(cfg, data)
            except Exception as _err:
                logger.warning(f"aupdate_state warning (non-fatal): {_err}")

    # 1. Try Redis cache
    redis_c = get_redis_client()
    if redis_c:
        try:
            cached_json = await redis_c.get(f"diag_state:{session_id}")
            if cached_json:
                state_data = json.loads(cached_json)
                await _safe_update_graph(thread_config, state_data)
                logger.info(f"[restore_state] Restored session {session_id} from Redis cache.")
                return state_data
        except Exception as re_err:
            logger.warning(f"Redis state lookup error: {re_err}")

    # 2. Try MongoDB persistence
    if mongo_db:
        try:
            col = mongo_db.get_collection("diagnostic_states")
            doc = await col.find_one({"_id": session_id})
            if doc and "state" in doc and doc["state"]:
                state_data = doc["state"]
                await _safe_update_graph(thread_config, state_data)
                logger.info(f"[restore_state] Restored session {session_id} from MongoDB.")
                return state_data
        except Exception as mg_err:
            logger.warning(f"MongoDB state lookup error: {mg_err}")

    # 3. Fallback: Restore from PostgreSQL DiagnosticSession
    try:
        target_uuid = uuid.UUID(session_id)
    except (ValueError, TypeError):
        return None

    stmt = select(DiagnosticSession).where(DiagnosticSession.id == target_uuid)
    res = await db.execute(stmt)
    session_record = res.scalar_one_or_none()
    if not session_record:
        return None

    logger.info(f"[restore_state] Reconstructing state for session {session_id} from PostgreSQL record.")
    raw_topic = session_record.topic
    kp = dict(session_record.knowledge_profile or {})
    turns_used = session_record.turns_used or 0

    # Retrieve checklist
    checklist_items = await get_diagnostic_checklist(
        topic=raw_topic,
        mongo_db=mongo_db,
        user_id=session_record.user_id
    )
    concept_checklist = [item["concept_id"] for item in checklist_items]
    if not kp:
        kp = {c: {"confidence": 0.0, "status": "unassessed"} for c in concept_checklist}

    # Find the first unassessed concept
    unassessed = [c for c in concept_checklist if kp.get(c, {}).get("status") == "unassessed"]
    target_concept = unassessed[0] if unassessed else (concept_checklist[0] if concept_checklist else "fundamentals")
    target_title = target_concept.replace("_", " ").title()

    # Retrieve cached question or fallback
    topic_slug = raw_topic.strip().lower().replace(" ", "_").replace("-", "_")
    cache_key = f"diag_q:{topic_slug}:{target_concept}"
    q_str = None
    if redis_c:
        try:
            q_str = await redis_c.get(cache_key)
        except Exception:
            pass
    if not q_str and mongo_db:
        try:
            q_doc = await mongo_db.get_collection("diagnostic_questions").find_one({"_id": cache_key})
            if q_doc and "question" in q_doc:
                q_str = q_doc["question"]
        except Exception:
            pass
    if not q_str:
        q_str = f"Describe your approach, mechanism, and time/space complexity for **{target_title}** in `{raw_topic}`."

    # Build reconstructed state
    reconstructed_state = {
        "session_id": session_id,
        "user_id": str(session_record.user_id),
        "topic": raw_topic,
        "concept_checklist": concept_checklist,
        "knowledge_profile": kp,
        "current_concept": target_concept,
        "current_question": q_str,
        "last_user_answer": None,
        "turns_used": turns_used,
        "is_complete": session_record.completed_at is not None,
        "messages": [{
            "id": f"m_{turns_used}",
            "sender": "ai",
            "text": q_str,
            "timestamp": datetime.now(timezone.utc).strftime("%I:%M %p")
        }],
        "turn_history": []
    }

    # Hydrate LangGraph state!
    await _safe_update_graph(thread_config, reconstructed_state)

    # Persist into Redis and Mongo
    await persist_diagnostic_state(session_id, reconstructed_state, mongo_db)

    return reconstructed_state


@router.post("/start", status_code=status.HTTP_200_OK)
async def start_diagnostic(
    payload: StartDiagnosticSchema,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    mongo_db: Any = Depends(get_mongo_db),
    checkpointer: Any = Depends(get_redis_checkpointer)
):
    """Initializes a new diagnostic session or resumes an existing active diagnostic session checkpointed in Redis, MongoDB, or Postgres."""
    import asyncio
    import time
    from sqlalchemy import or_, func
    t_start = time.monotonic()
    raw_topic = payload.topic.strip()
    logger.info(f"[start_diagnostic] ▶ START | topic='{raw_topic}' session_id='{payload.session_id}' user={current_user.id}")

    try:
        existing_session = None

        # Step 1A: If session_id is explicitly provided, look it up directly
        if payload.session_id:
            try:
                target_uuid = uuid.UUID(payload.session_id)
                stmt = (
                    select(DiagnosticSession)
                    .where(
                        DiagnosticSession.id == target_uuid,
                        DiagnosticSession.user_id == current_user.id
                    )
                )
                res = await db.execute(stmt)
                existing_session = res.scalar_one_or_none()
            except (ValueError, TypeError):
                existing_session = None

        # Step 1B: Check for existing incomplete session for this user and topic in PostgreSQL
        if not existing_session:
            topic_clean = raw_topic.lower().replace("_", " ").replace("-", " ")
            topic_slug = topic_clean.replace(" ", "_")
            topic_hyphen = topic_clean.replace(" ", "-")
            stmt = (
                select(DiagnosticSession)
                .where(
                    DiagnosticSession.user_id == current_user.id,
                    or_(
                        DiagnosticSession.topic == raw_topic,
                        func.lower(DiagnosticSession.topic) == raw_topic.lower(),
                        func.lower(func.replace(func.replace(DiagnosticSession.topic, "_", " "), "-", " ")) == topic_clean,
                        func.lower(func.replace(DiagnosticSession.topic, " ", "_")) == topic_slug,
                        func.lower(func.replace(DiagnosticSession.topic, " ", "-")) == topic_hyphen
                    ),
                    DiagnosticSession.completed_at.is_(None)
                )
                .order_by(DiagnosticSession.turns_used.desc())
            )
            res = await db.execute(stmt)
            existing_session = res.scalars().first()

        if existing_session:
            session_id = str(existing_session.id)
            state_vals = await get_or_restore_diagnostic_state(session_id, db, mongo_db, checkpointer)
            if not state_vals:
                state_vals = {}

            logger.info(f"[start_diagnostic] 🔄 RESUMING existing session {session_id} at turn {state_vals.get('turns_used', existing_session.turns_used)}")

            # Sync user_topics mode_choice in MongoDB
            if mongo_db:
                try:
                    from app.engine.curriculum_engine import get_or_create_user_topic
                    topic_doc = await get_or_create_user_topic(current_user.id, raw_topic, mongo_db)
                    ut_collection = mongo_db.get_collection("user_topics")
                    await ut_collection.update_one(
                        {"user_id": str(current_user.id), "topic_slug": topic_doc["topic_slug"]},
                        {"$set": {"mode_choice": "diagnostic", "diagnostic_session_id": session_id, "updated_at": datetime.now(timezone.utc).isoformat()}}
                    )
                except Exception as ut_err:
                    logger.warning(f"Failed to update user_topics mode_choice: {ut_err}")

            kp = state_vals.get("knowledge_profile") or existing_session.knowledge_profile or {}
            res_turn = state_vals.get("turns_used") or existing_session.turns_used or 1
            res_concept = state_vals.get("current_concept") or "fundamentals"
            res_title = res_concept.replace("_", " ").title()
            res_question = state_vals.get("current_question") or f"Explain your approach and complexity for **{res_title}**."

            messages = state_vals.get("messages", [])
            if not messages:
                messages = [{
                    "id": f"m_{res_turn}",
                    "sender": "ai",
                    "text": res_question,
                    "timestamp": datetime.now(timezone.utc).strftime("%I:%M %p")
                }]

            return {
                "session_id": session_id,
                "turn": res_turn,
                "current_concept": res_concept,
                "current_question": res_question,
                "is_complete": state_vals.get("is_complete", existing_session.completed_at is not None),
                "knowledge_profile": kp,
                "messages": messages,
                "turn_history": state_vals.get("turn_history", []),
                "is_resumed": True,
                "user_input": {
                    "input_id": f"diag_input_{session_id}_{res_turn}",
                    "type": "text",
                    "label": f"Concept Probe: {res_title}",
                    "description": "Explain your approach, mechanism, and time/space complexity.",
                    "required": True
                }
            }

        # Step 2: Create brand new diagnostic session
        checklist_items = await get_diagnostic_checklist(
            topic=raw_topic,
            mongo_db=mongo_db,
            user_id=current_user.id
        )

        concept_checklist = [item["concept_id"] for item in checklist_items]
        session_id = str(uuid.uuid4())
        thread_config = {"configurable": {"thread_id": session_id}}
        graph_app = build_diagnostic_graph(checkpointer)
        initial_knowledge_profile = {c: {"confidence": 0.0, "status": "unassessed"} for c in concept_checklist}

        initial_state = {
            "session_id": session_id,
            "user_id": str(current_user.id),
            "topic": raw_topic,
            "concept_checklist": concept_checklist,
            "knowledge_profile": initial_knowledge_profile,
            "current_concept": None,
            "current_question": None,
            "last_user_answer": None,
            "turns_used": 0,
            "is_complete": False,
            "messages": [],
            "turn_history": []
        }

        new_session = DiagnosticSession(
            id=uuid.UUID(session_id),
            user_id=current_user.id,
            topic=raw_topic,
            knowledge_profile=initial_knowledge_profile,
            turns_used=0
        )
        db.add(new_session)

        # Run DB commit & LangGraph first-question invocation concurrently
        t2 = time.monotonic()
        db_commit_task = asyncio.create_task(db.commit())
        graph_invoke_task = asyncio.create_task(
            graph_app.ainvoke(initial_state, config=thread_config)
        )
        results = await asyncio.gather(db_commit_task, graph_invoke_task, return_exceptions=True)
        logger.info(f"[start_diagnostic] ⏱ db+graph_invoke: {int((time.monotonic()-t2)*1000)}ms")

        if isinstance(results[0], Exception):
            logger.warning(f"[start_diagnostic] DB commit failed (non-fatal): {results[0]}")
        if isinstance(results[1], Exception):
            raise results[1]

        state = results[1]

        # Sync user_topics mode_choice in MongoDB
        if mongo_db:
            try:
                from app.engine.curriculum_engine import get_or_create_user_topic
                topic_doc = await get_or_create_user_topic(current_user.id, raw_topic, mongo_db)
                ut_collection = mongo_db.get_collection("user_topics")
                await ut_collection.update_one(
                    {"user_id": str(current_user.id), "topic_slug": topic_doc["topic_slug"]},
                    {"$set": {"mode_choice": "diagnostic", "diagnostic_session_id": session_id, "updated_at": datetime.now(timezone.utc).isoformat()}}
                )
            except Exception as ut_err:
                logger.warning(f"Failed to update user_topics mode_choice: {ut_err}")

        new_concept = state.get("current_concept") or "fundamentals"
        new_title = new_concept.replace("_", " ").title()
        new_turn = state.get("turns_used", 1)
        return {
            "session_id": session_id,
            "turn": new_turn,
            "current_concept": new_concept,
            "current_question": state.get("current_question"),
            "is_complete": False,
            "knowledge_profile": state.get("knowledge_profile", initial_knowledge_profile),
            "messages": state.get("messages", []),
            "turn_history": state.get("turn_history", []),
            "is_resumed": False,
            "user_input": {
                "input_id": f"diag_input_{session_id}_{new_turn}",
                "type": "text",
                "label": f"Concept Probe: {new_title}",
                "description": "Explain your approach, mechanism, and time/space complexity.",
                "required": True
            }
        }
    except Exception as e:
        logger.error(f"[start_diagnostic] Error for topic='{raw_topic}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to start diagnostic: {str(e)}")




from fastapi.responses import StreamingResponse
import json

@router.post("/answer")
async def answer_diagnostic(
    payload: AnswerDiagnosticSchema,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    mongo_db: Any = Depends(get_mongo_db),
    checkpointer: Any = Depends(get_redis_checkpointer)
):
    """Resumes the LangGraph thread, evaluates answer, and returns next question or final profile."""
    thread_config = {"configurable": {"thread_id": payload.session_id}}
    graph_app = build_diagnostic_graph(checkpointer)
    
    current_state_vals = await get_or_restore_diagnostic_state(payload.session_id, db, mongo_db, checkpointer)
    if not current_state_vals:
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

    # Persist state to Redis & MongoDB for durability
    await persist_diagnostic_state(payload.session_id, updated_state, mongo_db)

    evaluated_concept = (
        updated_state.get('last_evaluated_concept')
        or current_state_vals.get('current_concept')
    )
    concept_feedback = (
        updated_state.get('last_evaluated_feedback')
        or (profile.get(evaluated_concept, {}).get('feedback', '') if evaluated_concept else '')
    )

    if is_complete:
        return {
            "session_id": payload.session_id,
            "turn": turns_used,
            "is_complete": True,
            "knowledge_profile": profile,
            "feedback": concept_feedback,
            "last_evaluated_concept": evaluated_concept,
            "turn_history": updated_state.get("turn_history", [])
        }

    ans_concept = updated_state.get("current_concept") or "fundamentals"
    ans_title = ans_concept.replace("_", " ").title()
    return {
        "session_id": payload.session_id,
        "turn": turns_used,
        "current_concept": ans_concept,
        "current_question": updated_state.get("current_question"),
        "is_complete": False,
        "knowledge_profile": profile,
        "feedback": concept_feedback,
        "last_evaluated_concept": evaluated_concept,
        "turn_history": updated_state.get("turn_history", []),
        "user_input": {
            "input_id": f"diag_input_{payload.session_id}_{turns_used}",
            "type": "text",
            "label": f"Concept Probe: {ans_title}",
            "description": "Explain your approach, mechanism, and time/space complexity.",
            "required": True
        }
    }


async def stream_diagnostic_answer_generator(
    payload: AnswerDiagnosticSchema,
    db: AsyncSession,
    checkpointer: Any,
    mongo_db: Any = None
):
    thread_config = {"configurable": {"thread_id": payload.session_id}}
    graph_app = build_diagnostic_graph(checkpointer)

    current_state_vals = await get_or_restore_diagnostic_state(payload.session_id, db, mongo_db, checkpointer)
    if not current_state_vals:
        err_event = ErrorEvent(message="Session expired or not found.", code="SESSION_NOT_FOUND")
        yield format_sse(err_event)
        yield f"data: {json.dumps({'event': 'error', 'message': 'Session expired or not found.'})}\n\n"
        return

    # Validate input value if input_id was supplied
    if payload.input_id:
        temp_input = UserInputPayload(
            input_id=payload.input_id,
            type="text",
            label="Diagnostic Answer",
            required=True,
            validation=ValidationConstraints(min_length=1)
        )
        is_val, err_msg = validate_user_input_submission(temp_input, payload.answer)
        if not is_val:
            err_event = ErrorEvent(message=err_msg or "Invalid answer submission.", code="VALIDATION_ERROR")
            yield format_sse(err_event)
            yield f"data: {json.dumps({'event': 'error', 'message': err_msg})}\n\n"
            return

    # Emit message_start event
    yield format_sse(MessageStartEvent(
        session_id=payload.session_id,
        role="assistant",
        metadata={"turn": current_state_vals.get("turns_used", 0) + 1}
    ))

    # 1. Resume graph execution to evaluate answer (score_node)
    yield format_sse(ToolStartEvent(
        tool_name="score_node",
        tool_input={"session_id": payload.session_id, "answer": payload.answer}
    ))

    try:
        updated_state = await graph_app.ainvoke(
            {"last_user_answer": payload.answer},
            config=thread_config
        )
    except Exception as exc:
        logger.error(f"[stream_diagnostic_answer_generator] Graph execution failed: {exc}", exc_info=True)
        yield format_sse(ErrorEvent(message=f"Evaluation error: {str(exc)}", code="GRAPH_EXECUTION_ERROR"))
        yield f"data: {json.dumps({'event': 'error', 'message': str(exc)})}\n\n"
        return

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

    # Persist state to Redis & MongoDB for durability
    await persist_diagnostic_state(payload.session_id, updated_state, mongo_db)

    # Emit evaluation tool completion & evaluation events
    evaluated_concept = (
        updated_state.get('last_evaluated_concept')
        or current_state_vals.get('current_concept')
        or updated_state.get('current_concept')
    )
    concept_feedback = (
        updated_state.get('last_evaluated_feedback')
        or (profile.get(evaluated_concept, {}).get('feedback', '') if evaluated_concept else '')
    )
    
    yield format_sse(ToolEndEvent(
        tool_name="score_node",
        tool_output={"concept": evaluated_concept, "feedback": concept_feedback, "turns_used": turns_used}
    ))
    # Legacy event for backward compatibility
    yield f"data: {json.dumps({'event': 'evaluated', 'turn': turns_used, 'is_complete': is_complete, 'knowledge_profile': profile, 'current_concept': evaluated_concept, 'feedback': concept_feedback})}\n\n"

    if is_complete:
        comp_md = "🎉 **Diagnostic Assessment Completed!**\n\nAll concepts have been evaluated. Your personalized curriculum is ready."
        yield format_sse(TextDeltaEvent(delta=comp_md))
        yield format_sse(MessageEndEvent(
            status="completed",
            session_id=payload.session_id,
            payload={
                "turn": turns_used,
                "is_complete": True,
                "knowledge_profile": profile,
                "feedback": concept_feedback,
                "last_evaluated_concept": evaluated_concept,
                "turn_history": updated_state.get("turn_history", [])
            }
        ))
        yield f"data: {json.dumps({'event': 'complete', 'session_id': payload.session_id, 'turn': turns_used, 'is_complete': True, 'knowledge_profile': profile, 'feedback': concept_feedback, 'last_evaluated_concept': evaluated_concept, 'turn_history': updated_state.get('turn_history', [])})}\n\n"
        return

    # Stream text deltas for next diagnostic question
    from app.llm.gateway import call_llm_gateway_stream, parse_json_from_llm_response
    from app.llm.prompts import DIAGNOSTIC_ASK_PROMPT
    from app.llm.schemas import DiagnosticQuestionSchema
    from app.db.redis_client import get_redis_client
    from app.db.mongo import get_mongo_db

    target_concept = updated_state.get("current_concept") or "fundamentals"
    raw_topic = updated_state.get("topic", "")
    topic_slug = raw_topic.strip().lower().replace(" ", "_")
    cache_key = f"diag_q:{topic_slug}:{target_concept}"

    redis_c = get_redis_client()
    mongo_db = get_mongo_db()

    # Check cache
    cached_q = None
    if redis_c:
        try:
            cached_q = await redis_c.get(cache_key)
        except Exception:
            pass

    if not cached_q and mongo_db:
        try:
            col = mongo_db.get_collection("diagnostic_questions")
            existing_doc = await col.find_one({"_id": cache_key})
            if existing_doc and "question" in existing_doc:
                cached_q = existing_doc["question"]
        except Exception:
            pass

    q_title = target_concept.replace('_', ' ').title()
    input_id = f"diag_input_{payload.session_id}_{turns_used}"

    if cached_q:
        q_md = f"### 💡 Concept Probe: `{q_title}`\n\n{cached_q}"
        yield format_sse(TextDeltaEvent(delta=q_md))
        
        # Emit structured user_input event
        u_input = UserInputPayload(
            input_id=input_id,
            type="text",
            label=f"Concept Probe: {q_title}",
            description="Explain your approach, mechanism, and time/space complexity.",
            required=True,
            validation=ValidationConstraints(min_length=3),
            metadata={"session_id": payload.session_id, "concept": target_concept, "turn": turns_used}
        )
        yield format_sse(UserInputEvent(input=u_input))
        yield format_sse(MessageEndEvent(
            status="awaiting_user",
            session_id=payload.session_id,
            payload={
                "turn": turns_used,
                "current_concept": target_concept,
                "current_question": cached_q,
                "is_complete": False,
                "knowledge_profile": profile,
                "input_id": input_id,
                "feedback": concept_feedback,
                "last_evaluated_concept": evaluated_concept,
                "turn_history": updated_state.get("turn_history", []),
            }
        ))
        yield f"data: {json.dumps({'event': 'question_complete', 'current_concept': target_concept, 'current_question': cached_q, 'markdown': q_md, 'turn': turns_used, 'input_id': input_id, 'feedback': concept_feedback, 'last_evaluated_concept': evaluated_concept})}\n\n"
        return

    # Stream question generation tokens live
    yield format_sse(ToolStartEvent(
        tool_name="ask_node",
        tool_input={"concept": target_concept, "topic": raw_topic, "turn": turns_used}
    ))
    yield f"data: {json.dumps({'event': 'question_start', 'current_concept': target_concept})}\n\n"
    q_tokens = []
    try:
        header_md = f"### 💡 Concept Probe: `{q_title}`\n\n"
        yield format_sse(TextDeltaEvent(delta=header_md))

        async for chunk_text in call_llm_gateway_stream(
            DIAGNOSTIC_ASK_PROMPT,
            {"concept": target_concept, "topic": raw_topic, "history": turns_used},
            DiagnosticQuestionSchema
        ):
            q_tokens.append(chunk_text)
            # Legacy delta event
            yield f"data: {json.dumps({'event': 'question_token', 'delta': chunk_text})}\n\n"

        raw_q_json = "".join(q_tokens)
        parsed = parse_json_from_llm_response(raw_q_json, DiagnosticQuestionSchema, topic_hint=target_concept)
        question_str = getattr(parsed, "question", str(parsed))

        # Stream full formatted question text delta if not already streamed
        yield format_sse(TextDeltaEvent(delta=question_str))

        yield format_sse(ToolEndEvent(
            tool_name="ask_node",
            tool_output={"concept": target_concept, "question": question_str}
        ))

        # Cache question
        if redis_c and question_str:
            try:
                await redis_c.set(cache_key, question_str, ex=86400 * 30)
            except Exception:
                pass
        if mongo_db and question_str:
            try:
                col = mongo_db.get_collection("diagnostic_questions")
                await col.update_one(
                    {"_id": cache_key},
                    {"$set": {"_id": cache_key, "topic": topic_slug, "concept": target_concept, "question": question_str}},
                    upsert=True
                )
            except Exception:
                pass

        q_md = f"### 💡 Concept Probe: `{q_title}`\n\n{question_str}"

        # Emit structured user_input event
        u_input = UserInputPayload(
            input_id=input_id,
            type="text",
            label=f"Concept Probe: {q_title}",
            description="Explain your approach, mechanism, and time/space complexity.",
            required=True,
            validation=ValidationConstraints(min_length=3),
            metadata={"session_id": payload.session_id, "concept": target_concept, "turn": turns_used}
        )
        yield format_sse(UserInputEvent(input=u_input))
        yield format_sse(MessageEndEvent(
            status="awaiting_user",
            session_id=payload.session_id,
            payload={
                "turn": turns_used,
                "current_concept": target_concept,
                "current_question": question_str,
                "is_complete": False,
                "knowledge_profile": profile,
                "input_id": input_id,
                "feedback": concept_feedback,
                "last_evaluated_concept": evaluated_concept,
                "turn_history": updated_state.get("turn_history", []),
            }
        ))
        yield f"data: {json.dumps({'event': 'question_complete', 'current_concept': target_concept, 'current_question': question_str, 'markdown': q_md, 'turn': turns_used, 'input_id': input_id, 'feedback': concept_feedback, 'last_evaluated_concept': evaluated_concept})}\n\n"
    except Exception as err:
        logger.error(f"[stream_diagnostic_answer_generator] Error: {err}", exc_info=True)
        yield format_sse(ErrorEvent(message=str(err), code="DIAGNOSTIC_STREAM_ERROR"))
        yield f"data: {json.dumps({'event': 'error', 'message': str(err)})}\n\n"


@router.post("/answer/stream")
async def answer_diagnostic_stream(
    payload: AnswerDiagnosticSchema,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    mongo_db: Any = Depends(get_mongo_db),
    checkpointer: Any = Depends(get_redis_checkpointer)
):
    """
    Server-Sent Events (SSE) streaming endpoint for diagnostic chat turns.
    Evaluates user answer and streams the next diagnostic question text chunk-by-chunk.
    """
    return StreamingResponse(
        stream_diagnostic_answer_generator(payload, db, checkpointer, mongo_db),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

@router.post("/opt-out")
async def opt_out(
    payload: OptOutDiagnosticSchema,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    mongo_db: Any = Depends(get_mongo_db),
    checkpointer: Any = Depends(get_redis_checkpointer)
):
    """Skips all 15 diagnostic turns and marks all concepts unknown for beginners."""
    thread_config = {"configurable": {"thread_id": payload.session_id}}
    graph_app = build_diagnostic_graph(checkpointer)
    
    current_state_vals = await get_or_restore_diagnostic_state(payload.session_id, db, mongo_db, checkpointer)
    if not current_state_vals:
        raise HTTPException(status_code=404, detail="Session not found.")

    checklist = current_state_vals.get("concept_checklist", [])
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

@router.get("/active/{topic}")
async def get_active_diagnostic_session(
    topic: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    mongo_db: Any = Depends(get_mongo_db),
    checkpointer: Any = Depends(get_redis_checkpointer)
):
    """Retrieves active incomplete diagnostic session for the user and topic."""
    raw_topic = topic.strip()
    topic_clean = raw_topic.lower().replace("_", " ").replace("-", " ")
    topic_slug = topic_clean.replace(" ", "_")
    topic_hyphen = topic_clean.replace(" ", "-")
    from sqlalchemy import or_, func
    stmt = (
        select(DiagnosticSession)
        .where(
            DiagnosticSession.user_id == current_user.id,
            or_(
                DiagnosticSession.topic == raw_topic,
                func.lower(DiagnosticSession.topic) == raw_topic.lower(),
                func.lower(func.replace(func.replace(DiagnosticSession.topic, "_", " "), "-", " ")) == topic_clean,
                func.lower(func.replace(DiagnosticSession.topic, " ", "_")) == topic_slug,
                func.lower(func.replace(DiagnosticSession.topic, " ", "-")) == topic_hyphen
            ),
            DiagnosticSession.completed_at.is_(None)
        )
        .order_by(DiagnosticSession.turns_used.desc())
    )
    res = await db.execute(stmt)
    session_record = res.scalars().first()

    if not session_record:
        raise HTTPException(status_code=404, detail=f"No active diagnostic session for topic '{raw_topic}'.")

    session_id = str(session_record.id)
    state_vals = await get_or_restore_diagnostic_state(session_id, db, mongo_db, checkpointer)
    if not state_vals:
        state_vals = {}

    return {
        "session_id": session_id,
        "user_id": str(session_record.user_id),
        "topic": session_record.topic,
        "turns_used": state_vals.get("turns_used", session_record.turns_used),
        "current_concept": state_vals.get("current_concept"),
        "current_question": state_vals.get("current_question"),
        "is_complete": state_vals.get("is_complete", session_record.completed_at is not None),
        "knowledge_profile": state_vals.get("knowledge_profile", session_record.knowledge_profile),
        "messages": state_vals.get("messages", []),
        "turn_history": state_vals.get("turn_history", []),
        "completed_at": session_record.completed_at
    }


@router.get("/{session_id}")
async def get_diagnostic_state(
    session_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    mongo_db: Any = Depends(get_mongo_db),
    checkpointer: Any = Depends(get_redis_checkpointer)
):
    """Retrieves state and chat history of a diagnostic session."""
    try:
        session_uuid = uuid.UUID(session_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid session_id format.")

    stmt = select(DiagnosticSession).where(DiagnosticSession.id == session_uuid)
    res = await db.execute(stmt)
    session_record = res.scalar_one_or_none()

    if not session_record:
        raise HTTPException(status_code=404, detail="Diagnostic session not found.")

    state_vals = await get_or_restore_diagnostic_state(session_id, db, mongo_db, checkpointer)
    if not state_vals:
        state_vals = {}

    return {
        "session_id": str(session_record.id),
        "user_id": str(session_record.user_id),
        "topic": session_record.topic,
        "turns_used": state_vals.get("turns_used", session_record.turns_used),
        "current_concept": state_vals.get("current_concept"),
        "current_question": state_vals.get("current_question"),
        "is_complete": state_vals.get("is_complete", session_record.completed_at is not None),
        "knowledge_profile": state_vals.get("knowledge_profile", session_record.knowledge_profile),
        "messages": state_vals.get("messages", []),
        "turn_history": state_vals.get("turn_history", []),
        "completed_at": session_record.completed_at
    }