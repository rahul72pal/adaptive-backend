from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END
from app.llm.gateway import call_llm_gateway
from app.llm.prompts import RETEACH_LESSON_PROMPT
from app.llm.schemas import LessonExplanationSchema

class ReteachState(TypedDict):
    user_id: str
    concept_id: str
    topic: str
    strategies_tried: List[str]  # ["analogy", "visual_diagram", "interactive_exercise", "worked_example"]
    current_strategy: Optional[str]
    explanation_content: Optional[str]
    consecutive_failures: int
    action: str  # "continue" | "spot_check_prereq" | "re_diagnose"

async def select_strategy_node(state: ReteachState) -> ReteachState:
    """Selects an unused explanation strategy or triggers prerequisite fallback."""
    all_strategies = ["analogy", "visual_diagram", "worked_example", "interactive_exercise"]
    remaining = [s for s in all_strategies if s not in state.get("strategies_tried", [])]

    if not remaining or state.get("consecutive_failures", 0) >= 3:
        return {
            **state,
            "action": "spot_check_prereq"
        }

    next_strategy = remaining[0]
    topic_raw = state.get("topic") or "DSA"
    concept_id_raw = state.get("concept_id") or "concept"
    from app.catalog.context import TopicGenerationContext
    context = TopicGenerationContext.from_topic_identifier(topic_raw)

    res = await call_llm_gateway(
        RETEACH_LESSON_PROMPT,
        {
            "concept_id": concept_id_raw,
            "topic": context.canonical_topic,
            "strategies_tried": state.get("strategies_tried", []),
            "strategy": next_strategy
        },
        LessonExplanationSchema
    )

    updated_strategies = list(state.get("strategies_tried", [])) + [next_strategy]
    return {
        **state,
        "current_strategy": next_strategy,
        "strategies_tried": updated_strategies,
        "explanation_content": res.content,
        "action": "continue"
    }

async def fallback_prereq_node(state: ReteachState) -> ReteachState:
    """Triggers remediation regression for prerequisites when consecutive failures threshold is hit."""
    cid = state.get("concept_id") or "this concept"
    return {
        **state,
        "explanation_content": f"You've tried multiple explanation strategies for '{cid}'. Let's revisit foundational prerequisites first to build a solid base.",
        "action": "spot_check_prereq"
    }

def route_reteach(state: ReteachState) -> str:
    """Routes to prerequisite fallback or finishes re-teach node."""
    if state.get("action") == "spot_check_prereq":
        return "fallback_prereq_node"
    return END

# Build Reteach StateGraph
builder = StateGraph(ReteachState)
builder.add_node("select_strategy_node", select_strategy_node)
builder.add_node("fallback_prereq_node", fallback_prereq_node)

builder.set_entry_point("select_strategy_node")
builder.add_conditional_edges("select_strategy_node", route_reteach)
builder.add_edge("fallback_prereq_node", END)

reteach_graph_app = builder.compile()
