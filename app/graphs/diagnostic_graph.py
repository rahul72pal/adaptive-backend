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
    topic_slug = raw_topic.strip().lower().replace(" ", "_")
    cache_key = f"diag_q:{topic_slug}:{target_concept}"

    # 1. Check Redis Cache (< 5ms response time)
    redis_c = get_redis_client()
    if redis_c:
        try:
            cached_q = await redis_c.get(cache_key)
            if cached_q:
                logger.info(f"[ask_node] Redis Cache HIT for diagnostic question '{cache_key}'")
                return {
                    "current_concept": target_concept,
                    "current_question": cached_q,
                    "last_user_answer": None,
                    "turns_used": current_turn
                }
        except Exception as re_err:
            logger.warning(f"Redis lookup failed for key '{cache_key}': {re_err}")

    # 2. Check MongoDB persistence cache
    mongo_db = get_mongo_db()
    if mongo_db:
        try:
            col = mongo_db.get_collection("diagnostic_questions")
            existing_doc = await col.find_one({"_id": cache_key})
            if existing_doc and "question" in existing_doc:
                cached_q = existing_doc["question"]
                logger.info(f"[ask_node] MongoDB Cache HIT for diagnostic question '{cache_key}'")
                if redis_c:
                    try:
                        await redis_c.set(cache_key, cached_q, ex=86400 * 30)
                    except Exception:
                        pass
                return {
                    "current_concept": target_concept,
                    "current_question": cached_q,
                    "last_user_answer": None,
                    "turns_used": current_turn
                }
        except Exception as mg_err:
            logger.warning(f"MongoDB question lookup failed: {mg_err}")

    # 3. Cache MISS -> Call LLM Gateway
    res = await call_llm_gateway(
        DIAGNOSTIC_ASK_PROMPT,
        {
            "concept": target_concept,
            "topic": raw_topic,
            "history": current_turn
        },
        DiagnosticQuestionSchema
    )
    
    question_str = getattr(res, "question", str(res))

    # 4. Save to Redis & MongoDB cache for future diagnostic turns/sessions
    if redis_c and question_str:
        try:
            await redis_c.set(cache_key, question_str, ex=86400 * 30)
        except Exception as save_err:
            logger.warning(f"Failed to cache diagnostic question in Redis: {save_err}")

    if mongo_db and question_str:
        try:
            col = mongo_db.get_collection("diagnostic_questions")
            await col.update_one(
                {"_id": cache_key},
                {"$set": {"_id": cache_key, "topic": topic_slug, "concept": target_concept, "question": question_str}},
                upsert=True
            )
        except Exception as save_mg_err:
            logger.warning(f"Failed to cache diagnostic question in MongoDB: {save_mg_err}")

    return {
        "current_concept": target_concept,
        "current_question": question_str,
        "last_user_answer": None,
        "turns_used": current_turn
    }

async def score_node(state: DiagnosticState) -> Dict[str, Any]:
    """Evaluates the submitted answer and updates the confidence profile."""
    current_concept = state.get("current_concept") or "fundamentals"
    user_answer = state.get("last_user_answer") or ""

    res = await call_llm_gateway(
        DIAGNOSTIC_SCORE_PROMPT,
        {
            "answer": user_answer,
            "concept": current_concept,
            "question": state.get("current_question", "")
        },
        ScoreEvaluationSchema
    )

    updated_profile = dict(state.get("knowledge_profile", {}))
    updated_profile[current_concept] = {
        "confidence": res.confidence,
        "status": res.status,
        "feedback": getattr(res, "feedback", "")
    }

    return {"knowledge_profile": updated_profile}

def route_diagnostic(state: DiagnosticState) -> str:
    """Routes to finalize_node when all checklist concepts are assessed or max turns reached."""
    profile = state.get("knowledge_profile", {})
    checklist = state.get("concept_checklist", [])
    turns_used = state.get("turns_used", 0)

    unprobed = [c for c in checklist if profile.get(c, {}).get("status") == "unassessed"]

    if not unprobed or (checklist and turns_used >= len(checklist)) or turns_used >= 15:
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