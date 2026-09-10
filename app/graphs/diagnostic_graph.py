from typing import TypedDict, Optional, Dict, List, Any
from langgraph.graph import StateGraph, END
try:
    from langgraph.checkpoint.redis.aio import AsyncRedisSaver
except ImportError:
    AsyncRedisSaver = None

from app.llm.gateway import call_llm_gateway
from app.llm.prompts import DIAGNOSTIC_ASK_PROMPT, DIAGNOSTIC_SCORE_PROMPT
from app.llm.schemas import DiagnosticQuestionSchema, ScoreEvaluationSchema

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
    """Selects the next unprobed concept and generates a diagnostic question."""
    profile = state.get("knowledge_profile", {})
    checklist = state.get("concept_checklist", [])

    # Target concepts still at default confidence (0.5)
    unprobed = [c for c in checklist if profile.get(c, {}).get("confidence", 0.5) == 0.5]
    target_concept = unprobed[0] if unprobed else (checklist[0] if checklist else "fundamentals")
    
    current_turn = state.get("turns_used", 0) + 1
    
    res = await call_llm_gateway(
        DIAGNOSTIC_ASK_PROMPT,
        {
            "concept": target_concept,
            "topic": state.get("topic", ""),
            "history": current_turn
        },
        DiagnosticQuestionSchema
    )
    
    return {
        "current_concept": target_concept,
        "current_question": getattr(res, "question", str(res)),
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
    """Evaluates maximum turns for 15-turn diagnostic chat session."""
    if state.get("turns_used", 0) >= 15:
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