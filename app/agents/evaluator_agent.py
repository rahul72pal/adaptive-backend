import re
import logging
from typing import Dict, Any, List
from app.llm.gateway import call_llm_gateway
from app.llm.prompts import EVALUATOR_AGENT_PROMPT
from app.llm.schemas import EvaluatorGradeSchema

logger = logging.getLogger(__name__)

NON_ANSWER_REGEXES = [
    r"^\s*not\s+know\s*$",
    r"^\s*don'?t\s+know\s*$",
    r"^\s*i\s+don'?t\s+know\s*$",
    r"^\s*i\s+do\s+not\s+know\s*$",
    r"^\s*idk\s*$",
    r"^\s*no\s+idea\s*$",
    r"^\s*not\s+sure\s*$",
    r"^\s*no\s+clue\s*$",
    r"^\s*skip\s*$",
    r"^\s*pass\s*$",
    r"^\s*haven'?t\s+learned\s*$",
    r"^\s*dunno\s*$",
    r"^\s*nothing\s*$",
    r"^\s*no\s*$",
    r"^\s*na\s*$",
    r"^\s*n/a\s*$",
    r"^\s*\?\s*$",
]

COMPILED_NON_ANSWER = [re.compile(p, re.IGNORECASE) for p in NON_ANSWER_REGEXES]


class AnswerEvaluatorAgent:
    """Specialized Agent: Technical Assessment & Objective DSA Rubric Evaluator.
    
    Responsibilities:
    - Determine if answer is a non-answer (e.g. 'not know', 'idk', blank).
    - Objectively evaluate mechanism correctness and Big-O time/space complexity.
    - Output standardized EvaluatorGradeSchema without hallucinations or positive bias.
    """

    @classmethod
    def is_non_answer(cls, answer: str) -> bool:
        cleaned = (answer or "").strip()
        if not cleaned or len(cleaned) <= 1:
            return True
        for pattern in COMPILED_NON_ANSWER:
            if pattern.match(cleaned):
                return True
        return False

    @classmethod
    async def evaluate(cls, concept: str, question: str, answer: str) -> EvaluatorGradeSchema:
        raw_ans = (answer or "").strip()

        # 1. Deterministic fast-path: Handle explicit 'not know' or blank answers
        if cls.is_non_answer(raw_ans):
            logger.info(f"[AnswerEvaluatorAgent] Deterministic check: Non-answer detected for concept='{concept}' ('{raw_ans}'). Rating 0.0 unknown.")
            return EvaluatorGradeSchema(
                confidence=0.0,
                status="unknown",
                mechanism_correct=False,
                complexity_correct=False,
                key_gaps=["Student indicated no prior knowledge or approach for this concept."],
                evaluation_summary="Student stated they do not know the mechanism or complexity."
            )

        # 2. Substantive answers: Invoke LLM Evaluator Agent
        try:
            res: EvaluatorGradeSchema = await call_llm_gateway(
                EVALUATOR_AGENT_PROMPT,
                {
                    "concept": concept,
                    "question": question,
                    "answer": raw_ans
                },
                EvaluatorGradeSchema
            )
            logger.info(
                f"[AnswerEvaluatorAgent] Evaluated '{concept}': confidence={res.confidence}, "
                f"status={res.status}, mech={res.mechanism_correct}, compl={res.complexity_correct}"
            )
            return res
        except Exception as e:
            logger.error(f"[AnswerEvaluatorAgent] LLM Evaluation failed for concept '{concept}': {e}", exc_info=True)
            # Conservative fallback if LLM is temporarily unreachable
            return EvaluatorGradeSchema(
                confidence=0.3,
                status="unknown",
                mechanism_correct=False,
                complexity_correct=False,
                key_gaps=["Evaluation service encountered a fallback; manual check required."],
                evaluation_summary="Automated evaluation fallback triggered."
            )
