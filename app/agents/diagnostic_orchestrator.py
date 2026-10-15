import logging
from typing import Dict, Any, Optional
from app.agents.evaluator_agent import AnswerEvaluatorAgent
from app.agents.feedback_agent import FeedbackAgent
from app.llm.schemas import EvaluatorGradeSchema

logger = logging.getLogger(__name__)

class DiagnosticOrchestrator:
    """Multi-Agent Orchestrator for Diagnostic Assessment.
    
    Orchestration Pattern: Sequential Pipeline
    -----------------------------------------
    1. AnswerEvaluatorAgent:
       Evaluates the technical correctness of the student's mechanism and Big-O complexity.
       Applies deterministic guards for non-answers ('not know', 'idk', etc.) so no false
       positive scores or hallucinations occur.
    
    2. FeedbackAgent:
       Uses the evaluation grade, student's answer, and question to generate rich,
       supportive, pedagogical feedback in standard Markdown.
    
    3. Aggregator:
       Combines the signals into an updated confidence profile and turn history entry.
    """

    @classmethod
    async def process_turn(
        cls,
        concept: str,
        question: str,
        answer: str
    ) -> Dict[str, Any]:
        logger.info(f"[DiagnosticOrchestrator] Starting evaluation turn for concept='{concept}'")

        # Step 1: Run AnswerEvaluatorAgent
        grade: EvaluatorGradeSchema = await AnswerEvaluatorAgent.evaluate(
            concept=concept,
            question=question,
            answer=answer
        )
        logger.info(
            f"[DiagnosticOrchestrator] EvaluatorAgent output: confidence={grade.confidence}, status={grade.status}"
        )

        # Step 2: Run FeedbackAgent
        feedback_markdown: str = await FeedbackAgent.generate_feedback(
            concept=concept,
            question=question,
            answer=answer,
            grade=grade
        )
        logger.info(f"[DiagnosticOrchestrator] FeedbackAgent generated feedback ({len(feedback_markdown)} chars)")

        return {
            "concept": concept,
            "confidence": grade.confidence,
            "status": grade.status,
            "mechanism_correct": grade.mechanism_correct,
            "complexity_correct": grade.complexity_correct,
            "key_gaps": grade.key_gaps,
            "evaluation_summary": grade.evaluation_summary,
            "feedback": feedback_markdown
        }
