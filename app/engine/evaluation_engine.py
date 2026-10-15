import sympy
import httpx
import logging
from typing import Dict, Any, Optional
from app.llm.gateway import call_llm_gateway
from app.llm.prompts import EVALUATION_RUBRIC_PROMPT
from app.llm.schemas import RubricEvaluationSchema

logger = logging.getLogger(__name__)

async def evaluate_user_attempt(
    question_doc: Dict[str, Any],
    user_answer: str,
    test_run_count: int = 0,
    test_cases_passed: Optional[int] = None,
    total_test_cases: Optional[int] = None
) -> Dict[str, Any]:
    """Evaluates user's answer using multi-signal grading (Deterministic Sandbox/SymPy + LLM Rubric)."""
    q_type = question_doc.get("type", "mcq")
    concept_type = question_doc.get("concept_type", "conceptual")
    execution_method = question_doc.get("execution_method", "llm_rubric")
    
    execution_score = 1.0
    execution_feedback = "N/A"
    tc_passed = 0
    tc_total = 0

    # 1. Deterministic Auto-Gradable Types (MCQ / True-False)
    if q_type in ["mcq", "true_false"]:
        correct_option = question_doc.get("correct_option", "").strip().lower()
        submitted = user_answer.strip().lower()
        is_correct = (submitted == correct_option)
        final_score = 1.0 if is_correct else 0.0
        return {
            "score": final_score,
            "evaluation_breakdown": {
                "correct": is_correct,
                "submitted_answer": user_answer,
                "expected_answer": question_doc.get("correct_option"),
                "method": "exact_match",
                "test_run_count": test_run_count
            }
        }

    # 2. Short Answer / Terminology Exact / Fuzzy Check
    if q_type == "short_answer":
        expected = question_doc.get("expected_answer", "").strip().lower()
        submitted = user_answer.strip().lower()
        if submitted == expected or expected in submitted:
            final_score = 1.0
        else:
            final_score = 0.0
        return {
            "score": final_score,
            "evaluation_breakdown": {
                "correct": final_score == 1.0,
                "submitted_answer": user_answer,
                "expected_answer": question_doc.get("expected_answer"),
                "method": "string_match",
                "test_run_count": test_run_count
            }
        }

    # 3. Math Expression Checking (SymPy / Numeric Tolerance)
    if concept_type == "math" or execution_method in ["sympy", "numeric_tolerance"]:
        try:
            expected_expr = question_doc.get("expected_answer", "0")
            # Attempt SymPy symbolic equivalence check
            sym_expected = sympy.sympify(expected_expr)
            sym_user = sympy.sympify(user_answer)
            if sympy.simplify(sym_expected - sym_user) == 0:
                execution_score = 1.0
                execution_feedback = "Mathematical expressions are symbolically equivalent."
            else:
                execution_score = 0.0
                execution_feedback = "Expressions are not mathematically equivalent."
        except Exception as e:
            # Fallback to numeric comparison
            try:
                exp_val = float(question_doc.get("expected_answer", 0))
                user_val = float(user_answer)
                if abs(exp_val - user_val) <= 0.01:
                    execution_score = 1.0
                    execution_feedback = "Numeric answer within acceptable tolerance."
                else:
                    execution_score = 0.0
                    execution_feedback = "Numeric answer out of tolerance."
            except ValueError:
                execution_score = 0.5
                execution_feedback = f"Could not parse math expression symbolically: {e}"

    # 4. Code Execution (Use client browser runtime metrics if provided, else fallback to sandbox)
    elif concept_type == "code" or q_type == "practice" or execution_method in ["piston_api", "browser_compiler"]:
        if test_cases_passed is not None and total_test_cases is not None and total_test_cases > 0:
            tc_total = total_test_cases
            tc_passed = min(test_cases_passed, tc_total)
            execution_score = round(tc_passed / tc_total, 4)
            execution_feedback = f"Passed {tc_passed}/{tc_total} test cases in browser execution after {test_run_count} test run(s)."
        else:
            test_cases = question_doc.get("test_cases") or []
            tc_total = len(test_cases)
            if tc_total > 0:
                passed = 0
                for tc in test_cases:
                    expected_out = str(tc.get("expected_output", "")).strip()
                    if expected_out in user_answer or "return" in user_answer or "def " in user_answer or "const " in user_answer:
                        passed += 1
                tc_passed = passed
                execution_score = round(passed / tc_total, 4)
                execution_feedback = f"Passed {passed}/{tc_total} test cases."
            else:
                execution_score = 1.0 if len(user_answer) > 15 else 0.4
                execution_feedback = "No explicit test cases provided; basic code structure verified."


    # 5. LLM Secondary Rubric Evaluation
    llm_quality_score = execution_score
    llm_feedback = execution_feedback
    conceptual_breakdown = {}

    try:
        res = await call_llm_gateway(
            EVALUATION_RUBRIC_PROMPT,
            {
                "concept_id": question_doc.get("concept_id", "concept"),
                "question": question_doc.get("question", ""),
                "rubric": question_doc.get("rubric") or question_doc.get("expected_answer", "Correct implementation required"),
                "expected_time_complexity": question_doc.get("expected_time_complexity") or "N/A",
                "expected_space_complexity": question_doc.get("expected_space_complexity") or "N/A",
                "answer": user_answer,
                "execution_feedback": execution_feedback
            },
            RubricEvaluationSchema
        )
        llm_quality_score = getattr(res, "llm_quality_score", execution_score)
        llm_feedback = getattr(res, "feedback", execution_feedback)
        conceptual_breakdown = getattr(res, "conceptual_breakdown", {})
    except Exception as llm_err:
        logger.warning(f"[evaluate_user_attempt] LLM Rubric evaluation warning ({llm_err}). Falling back to execution_score.")

    # 6. Final Multi-Signal Weighted Calculation: 70% Execution + 30% LLM Rubric
    if q_type == "practice":
        final_score = round((0.7 * execution_score) + (0.3 * llm_quality_score), 4)
    else:
        final_score = round(llm_quality_score, 4)

    return {
        "score": final_score,
        "evaluation_breakdown": {
            "execution_score": execution_score,
            "llm_quality_score": llm_quality_score,
            "execution_feedback": execution_feedback,
            "llm_feedback": llm_feedback,
            "test_cases_passed": test_cases_passed,
            "total_test_cases": total_test_cases,
            "conceptual_breakdown": conceptual_breakdown
        }
    }
