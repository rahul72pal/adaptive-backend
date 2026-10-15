import logging
from typing import Dict, Any, Optional
from app.llm.gateway import call_llm_gateway
from app.llm.prompts import FEEDBACK_AGENT_PROMPT
from app.llm.schemas import FeedbackAgentSchema, EvaluatorGradeSchema

logger = logging.getLogger(__name__)

class FeedbackAgent:
    """Specialized Agent: Socratic DSA Feedback Specialist & Pedagogical Coach.
    
    Responsibilities:
    - Receive evaluation grade, question context, and student's answer.
    - Synthesize constructive, educational feedback formatted in Markdown.
    - Clearly explain mechanism, Big-O complexities, and key takeaways.
    - Provide immediate answers when student doesn't know, turning assessment into active learning.
    """

    @classmethod
    async def generate_feedback(
        cls,
        concept: str,
        question: str,
        answer: str,
        grade: EvaluatorGradeSchema
    ) -> str:
        concept_title = concept.replace("_", " ").title()
        raw_ans = (answer or "").strip()

        try:
            res: FeedbackAgentSchema = await call_llm_gateway(
                FEEDBACK_AGENT_PROMPT,
                {
                    "concept": concept_title,
                    "question": question,
                    "answer": raw_ans or "(No answer provided)",
                    "confidence": f"{grade.confidence:.2f}",
                    "status": grade.status,
                    "mechanism_correct": "Yes" if grade.mechanism_correct else "No",
                    "complexity_correct": "Yes" if grade.complexity_correct else "No",
                    "key_gaps": ", ".join(grade.key_gaps) if grade.key_gaps else "None identified",
                    "evaluation_summary": grade.evaluation_summary
                },
                FeedbackAgentSchema
            )
            feedback_text = getattr(res, "feedback_markdown", "") or ""
            if feedback_text.strip():
                return feedback_text
        except Exception as err:
            logger.error(f"[FeedbackAgent] Feedback generation LLM call failed for '{concept}': {err}", exc_info=True)

        # High-quality fallback if LLM is unreachable
        status_label = "Unassessed / Needs Practice" if grade.status == "unknown" else ("Partially Understood" if grade.status == "partial" else "Mastered")
        gaps_list = "\n".join([f"- {g}" for g in grade.key_gaps]) if grade.key_gaps else "- Continue practicing implementation variants."
        
        fallback_md = (
            f"**Assessment:** {status_label} for `{concept_title}`.\n\n"
            f"- **Observation:** {grade.evaluation_summary}\n"
            f"- **Identified Areas for Focus:**\n{gaps_list}\n\n"
            f"- **Core Mechanism & Complexity:** Make sure to study the fundamental operational trace and constant vs linear time/space bounds for this topic."
        )
        return fallback_md
