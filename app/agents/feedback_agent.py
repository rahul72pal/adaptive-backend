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

        # High-quality concise fallback if LLM is unreachable
        status_label = "Needs Practice" if grade.status == "unknown" else ("Partially Understood" if grade.status == "partial" else "Mastered")
        fallback_md = f"**{status_label}:** {grade.evaluation_summary or 'Response noted.'} Focus on standard operations and Big-O bounds for {concept_title}."
        return fallback_md
