import logging
from typing import TypedDict, Optional, Dict, List, Any
from langgraph.graph import StateGraph, END
try:
    from langgraph.checkpoint.redis.aio import AsyncRedisSaver
except ImportError:
    AsyncRedisSaver = None

from app.llm.gateway import call_llm_gateway
from app.llm.prompts import DIAGNOSTIC_ASK_PROMPT, DIAGNOSTIC_SCORE_PROMPT
from app.llm.schemas import DiagnosticQuestionSchema, ScoreEvaluationSchema
from app.db.redis_client import get_redis_client
from app.db.mongo import get_mongo_db

logger = logging.getLogger(__name__)

from datetime import datetime, timezone

class DiagnosticState(TypedDict):
    session_id: str
    user_id: str
    topic: str
    concept_checklist: List[str]
    knowledge_profile: Dict[str, Dict[str, Any]]
    current_concept: Optional[str]
    current_question: Optional[str]
    last_user_answer: Optional[str]
    turns_used: int
    is_complete: bool
    messages: List[Dict[str, Any]]
    turn_history: List[Dict[str, Any]]
    last_evaluated_concept: Optional[str]
    last_evaluated_feedback: Optional[str]

async def input_router(state: DiagnosticState) -> str:
    """Routes incoming HTTP execution based on whether an answer was provided."""
    if state.get("is_complete"):
        return END
    if state.get("last_user_answer"):
        return "score_node"
    return "ask_node"

async def ask_node(state: DiagnosticState) -> Dict[str, Any]:
    """Selects the next unprobed concept and retrieves or generates a diagnostic question with Redis & Mongo caching."""
    profile = state.get("knowledge_profile", {})
    checklist = state.get("concept_checklist", [])

    # Select the first concept that is still unassessed
    unprobed = [
        c for c in checklist 
        if profile.get(c, {}).get("status") == "unassessed"
    ]
    target_concept = unprobed[0] if unprobed else (checklist[0] if checklist else "fundamentals")
    
    current_turn = state.get("turns_used", 0) + 1
    raw_topic = state.get("topic", "")
    from app.catalog.context import TopicGenerationContext
    context = TopicGenerationContext.from_topic_identifier(raw_topic)
    cache_key = f"diag_q:{context.topic_id}:{target_concept}:{context.metadata_version}"

    question_str = None

    # 1. Check Redis Cache (< 5ms response time)
    redis_c = get_redis_client()
    if redis_c:
        try:
            cached_q = await redis_c.get(cache_key)
            if cached_q:
                logger.info(f"[ask_node] Redis Cache HIT for diagnostic question '{cache_key}'")
                question_str = cached_q
        except Exception as re_err:
            logger.warning(f"Redis lookup failed for key '{cache_key}': {re_err}")

    # 2. Check MongoDB persistence cache if not found in Redis
    if not question_str:
        mongo_db = get_mongo_db()
        if mongo_db:
            try:
                col = mongo_db.get_collection("diagnostic_questions")
                existing_doc = await col.find_one({"_id": cache_key})
                if existing_doc and "question" in existing_doc:
                    question_str = existing_doc["question"]
                    logger.info(f"[ask_node] MongoDB Cache HIT for diagnostic question '{cache_key}'")
                    if redis_c:
                        try:
                            await redis_c.set(cache_key, question_str, ex=86400 * 30)
                        except Exception:
                            pass
            except Exception as mg_err:
                logger.warning(f"MongoDB question lookup failed: {mg_err}")

    # 3. Cache MISS -> Call LLM Gateway
    if not question_str:
        try:
            res = await call_llm_gateway(
                DIAGNOSTIC_ASK_PROMPT,
                {
                    "concept": target_concept,
                    "topic": context.canonical_topic,
                    "topic_kind": context.kind,
                    "assessment_mode": context.assessment_mode,
                    "selected_language": context.selected_language or "python",
                    "history": current_turn
                },
                DiagnosticQuestionSchema
            )
            question_str = getattr(res, "question", str(res))
        except Exception as llm_err:
            logger.error(f"[ask_node] Failed to generate diagnostic question via LLM gateway: {llm_err}")
            concept_readable = target_concept.replace("_", " ").title()
            if context.assessment_mode == "conceptual_only":
                question_str = (
                    f"Explain the core principle, invariant, and mathematical reasoning for **{concept_readable}** in `{context.canonical_topic}`."
                )
            else:
                question_str = (
                    f"How would you approach solving or implementing **{concept_readable}** in `{context.canonical_topic}`? "
                    f"Walk through the core algorithmic mechanism, data structure choices, and time/space complexity."
                )

        # Save to Redis & MongoDB cache for future diagnostic turns/sessions
        if redis_c and question_str:
            try:
                await redis_c.set(cache_key, question_str, ex=86400 * 30)
            except Exception as save_err:
                logger.warning(f"Failed to cache diagnostic question in Redis: {save_err}")

        mongo_db = get_mongo_db()
        if mongo_db and question_str:
            try:
                col = mongo_db.get_collection("diagnostic_questions")
                await col.update_one(
                    {"_id": cache_key},
                    {
                        "$set": {
                            "_id": cache_key,
                            "topic_id": context.topic_id,
                            "topic": context.canonical_topic,
                            "concept": target_concept,
                            "question": question_str,
                            "metadata_version": context.metadata_version
                        }
                    },
                    upsert=True
                )
            except Exception as save_mg_err:
                logger.warning(f"Failed to cache diagnostic question in MongoDB: {save_mg_err}")

    # Append question to state messages list
    messages = list(state.get("messages") or [])
    messages.append({
        "id": f"m_{len(messages)}",
        "sender": "ai",
        "text": question_str,
        "timestamp": datetime.now(timezone.utc).strftime("%I:%M %p")
    })

    return {
        "current_concept": target_concept,
        "current_question": question_str,
        "last_user_answer": None,
        "turns_used": current_turn,
        "messages": messages
    }

from app.agents.diagnostic_orchestrator import DiagnosticOrchestrator

async def score_node(state: DiagnosticState) -> Dict[str, Any]:
    """Evaluates the submitted answer and generates rich feedback using the multi-agent orchestrator."""
    current_concept = state.get("current_concept") or "fundamentals"
    user_answer = state.get("last_user_answer") or ""
    current_question = state.get("current_question", "")
    raw_topic = state.get("topic", "")

    from app.catalog.context import TopicGenerationContext
    context = TopicGenerationContext.from_topic_identifier(raw_topic)

    # Execute multi-agent pipeline: EvaluatorAgent -> FeedbackAgent -> Aggregator
    result = await DiagnosticOrchestrator.process_turn(
        concept=current_concept,
        question=current_question,
        answer=user_answer,
        topic_kind=context.kind,
        assessment_mode=context.assessment_mode
    )

    confidence = result.get("confidence", 0.0)
    status_val = result.get("status", "unknown")
    feedback_text = result.get("feedback", "")

    messages = list(state.get("messages") or [])
    if user_answer:
        messages.append({
            "id": f"m_user_{len(messages)}",
            "sender": "user",
            "text": user_answer,
            "timestamp": datetime.now(timezone.utc).strftime("%I:%M %p")
        })

    turn_history = list(state.get("turn_history") or [])
    turn_history.append({
        "turn": state.get("turns_used", len(turn_history) + 1),
        "concept": current_concept,
        "question": current_question,
        "userAnswer": user_answer,
        "feedback": feedback_text or "Assessment recorded by Lumina AI engine."
    })

    updated_profile = dict(state.get("knowledge_profile", {}))
    updated_profile[current_concept] = {
        "confidence": confidence,
        "status": status_val,
        "feedback": feedback_text
    }

    return {
        "last_evaluated_concept": current_concept,
        "last_evaluated_feedback": feedback_text,
        "knowledge_profile": updated_profile,
        "messages": messages,
        "turn_history": turn_history
    }

def route_diagnostic(state: DiagnosticState) -> str:
    """Routes to finalize_node when all checklist concepts are assessed or max turns reached."""
    profile = state.get("knowledge_profile", {})
    checklist = state.get("concept_checklist", [])
    turns_used = state.get("turns_used", 0)

    unprobed = [c for c in checklist if profile.get(c, {}).get("status") == "unassessed"]
    max_turns = len(checklist) if checklist else 15

    if not unprobed or turns_used >= max_turns:
        return "finalize_node"
    return "ask_node"

async def finalize_node(state: DiagnosticState) -> Dict[str, Any]:
    """Marks the diagnostic flow complete."""
    return {"is_complete": True}

def build_diagnostic_graph(checkpointer: Any):
    builder = StateGraph(DiagnosticState)
    
    builder.add_node("ask_node", ask_node)
    builder.add_node("score_node", score_node)
    builder.add_node("finalize_node", finalize_node)

    builder.set_conditional_entry_point(
        input_router,
        {
            "ask_node": "ask_node",
            "score_node": "score_node",
            END: END
        }
    )
    
    # Suspends execution after generating question to wait for HTTP answer
    builder.add_edge("ask_node", END)
    
    # Routes after scoring: loop to ask next question, or finalize
    builder.add_conditional_edges(
        "score_node",
        route_diagnostic,
        {
            "ask_node": "ask_node",
            "finalize_node": "finalize_node"
        }
    )
    builder.add_edge("finalize_node", END)

    return builder.compile(checkpointer=checkpointer)