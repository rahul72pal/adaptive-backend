import uuid
from typing import Optional, Any, Dict, List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field

from app.db.postgres import get_db
from app.db.mongo import get_mongo_db
from app.core.dependencies import get_current_user
from app.db.models import User, Attempt, ConceptMastery
from app.engine.question_engine import get_or_generate_question, get_or_generate_question_batch
from app.engine.evaluation_engine import evaluate_user_attempt
from app.engine.mastery_engine import update_user_mastery_and_check_remediation
from app.engine.curriculum_engine import sync_learning_path_progress, slugify_topic, record_user_completed_node

router = APIRouter(prefix="/api", tags=["Question Engine & Evaluation"])

class GenerateQuestionSchema(BaseModel):
    topic: str
    concept_id: str
    concept_type: str = "conceptual"
    difficulty: int = 3
    language: Optional[str] = "python"
    explanation_language: Optional[str] = "english"

class GenerateQuestionBatchSchema(BaseModel):
    topic: str
    concept_id: str
    concept_type: str = "conceptual"
    difficulty: int = 3
    count: int = Field(default=3, ge=1, le=10)
    language: Optional[str] = "python"
    explanation_language: Optional[str] = "english"

class SubmitAttemptSchema(BaseModel):
    question_id: str
    concept_id: str
    topic: str
    difficulty: int = 1
    answer: str
    current_question_count: int = Field(default=1, ge=1)
    target_question_count: Optional[int] = Field(default=3, ge=1)
    test_run_count: Optional[int] = Field(default=0, ge=0)
    test_cases_passed: Optional[int] = Field(default=None, ge=0)
    total_test_cases: Optional[int] = Field(default=None, ge=0)
    language: Optional[str] = "python"
    explanation_language: Optional[str] = "english"
    include_next_question: bool = False


import json
import logging
import random
from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)

def _sse(payload: dict) -> str:
    return f"data: {json.dumps({k: v for k, v in payload.items() if v is not None})}\n\n"


async def stream_question_generator(payload: GenerateQuestionSchema, mongo_db: Any):
    """Split question pipeline as a tool-event stream.

    Events
    ------
    message_start → tool_start(question_text) → token* → tool_end
    → question_ready → [tool_start(code_and_tests) → meta → test_cases → tool_end]
    → complete | cached | error
    """
    from app.engine.question_engine import (
        build_question_doc,
        generate_question_meta,
        generate_test_cases,
        sanitize_question_for_client,
        question_kind_for,
        _find_cached_question,
    )
    from app.llm.gateway import call_llm_gateway_stream
    from app.llm.prompts import QUESTION_STATEMENT_PROMPT
    from app.llm.schemas import QuestionTextSchema

    topic_clean = (payload.topic or "").lower().strip()
    concept_id = (payload.concept_id or "").strip()
    concept_type = (payload.concept_type or "conceptual").strip().lower()
    difficulty = int(payload.difficulty or 3)
    language = (payload.language or "python").lower().strip()
    explanation_language = (payload.explanation_language or "english").lower().strip()
    question_kind = question_kind_for(concept_type)
    is_code = question_kind == "practice" or concept_type in ("code", "practice")

    q_collection = mongo_db.get_collection("questions") if mongo_db is not None else None

    # 1. Warm cache → instant response
    cached = await _find_cached_question(
        concept_id, topic_clean, difficulty, concept_type, q_collection
    )
    if cached:
        safe_q = sanitize_question_for_client(dict(cached))
        yield _sse({
            "event": "cached",
            "question": safe_q,
            "markdown": safe_q.get("markdown", ""),
            "tool": "question_text",
        })
        return

    yield _sse({"event": "message_start", "concept_id": concept_id, "difficulty": difficulty})

    # 2. Tool 1 — stream the statement only (fast, no JSON)
    yield _sse({"event": "tool_start", "tool": "question_text", "concept_id": concept_id})
    statement_chunks: List[str] = []
    stream_error: Optional[str] = None
    try:
        async for delta in call_llm_gateway_stream(
            QUESTION_STATEMENT_PROMPT,
            {
                "topic": topic_clean,
                "concept_id": concept_id,
                "concept_type": concept_type,
                "difficulty": difficulty,
                "question_type": question_kind,
                "language": language,
                "explanation_language": explanation_language,
            }
        ):
            statement_chunks.append(delta)
            yield _sse({"event": "token", "delta": delta})
    except Exception as stream_err:
        stream_error = str(stream_err) or stream_err.__class__.__name__
        logger.error(
            "[stream_question_generator] Statement stream failed for '%s': %s",
            concept_id, stream_error, exc_info=True,
        )

    statement = "".join(statement_chunks).strip()
    if not statement:
        # Never fake a statement: a placeholder would hide a real provider or
        # credential failure (bad/expired API key, provider that does not
        # stream) behind a question with no content. Report the reason instead
        # so the UI can show it; the client then falls back to the blocking
        # endpoint, which surfaces its own error if that also fails.
        yield _sse({
            "event": "error",
            "message": (
                f"Question generation failed: {stream_error}"
                if stream_error
                else "The model returned no content for this question."
            ),
        })
        return

    yield _sse({"event": "tool_end", "tool": "question_text"})
    yield _sse({
        "event": "question_ready",
        "question": {
            "concept_id": concept_id,
            "concept_type": concept_type,
            "difficulty": difficulty,
            "type": question_kind,
            "question": statement,
        },
        "markdown": statement,
    })

    # 3. Tool 2 — metadata (+ code spec) and test cases
    if is_code:
        yield _sse({"event": "tool_start", "tool": "code_and_tests", "concept_id": concept_id})

    meta: Optional[QuestionTextSchema] = None
    try:
        meta = await generate_question_meta(
            topic_clean, concept_id, concept_type, difficulty, language, statement
        )
        # Keep the streamed statement authoritative.
        meta.question = statement
    except Exception as meta_err:
        logger.warning("[stream_question_generator] Meta tool failed: %s", meta_err)
        meta = QuestionTextSchema(
            type="practice" if is_code else "mcq",
            concept_id=concept_id,
            difficulty=difficulty,
            question=statement,
            options=None if is_code else ["Option A", "Option B", "Option C", "Option D"],
            correct_option=None if is_code else "Option A",
            rubric=f"Evaluate correctness for {concept_id}." if is_code else None,
        )

    yield _sse({
        "event": "meta",
        "type": meta.type,
        "function_signature": meta.function_signature,
        "options": meta.options,
        "rubric": meta.rubric,
        "expected_time_complexity": meta.expected_time_complexity,
        "expected_space_complexity": meta.expected_space_complexity,
        "starter_code": meta.starter_code,
    })

    test_cases: List[Dict[str, Any]] = []
    if is_code:
        test_cases = await generate_test_cases(
            topic_clean,
            concept_id,
            statement,
            meta.function_signature,
            language,
        )
        yield _sse({
            "event": "test_cases",
            "test_cases": test_cases,
            "count": len(test_cases),
            "generated": bool(test_cases),
        })
        yield _sse({"event": "tool_end", "tool": "code_and_tests"})

    # 4. Assemble, persist, complete
    doc = build_question_doc(
        meta=meta,
        statement=statement,
        topic_clean=topic_clean,
        concept_id=concept_id,
        concept_type=concept_type,
        difficulty=difficulty,
        test_cases=test_cases,
        language=language,
    )

    if q_collection is not None:
        try:
            await q_collection.insert_one(dict(doc))
        except Exception as persist_err:
            logger.warning("[stream_question_generator] Persist failed: %s", persist_err)

    safe_q = sanitize_question_for_client(doc)
    yield _sse({
        "event": "complete",
        "question": safe_q,
        "markdown": safe_q.get("markdown", ""),
        "execution_method": safe_q.get("execution_method"),
        "has_test_cases": bool(safe_q.get("test_cases")),
    })


@router.post("/questions/generate")
async def generate_question(
    payload: GenerateQuestionSchema,
    current_user: User = Depends(get_current_user),
    mongo_db: Any = Depends(get_mongo_db)
):
    """Retrieves a single client-safe question for topic & concept_id from MongoDB cache or triggers mini-batch LLM generation."""
    try:
        question_doc = await get_or_generate_question(
            topic=payload.topic,
            concept_id=payload.concept_id,
            concept_type=payload.concept_type,
            difficulty=payload.difficulty,
            language=payload.language or "python",
            explanation_language=payload.explanation_language or "english",
            mongo_db=mongo_db
        )
        return question_doc
    except Exception as e:
        logger.error(f"[generate_question] Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Question generation failed: {str(e)}")


@router.post("/questions/generate/stream")
@router.post("/questions/stream")
async def generate_question_stream(
    payload: GenerateQuestionSchema,
    current_user: User = Depends(get_current_user),
    mongo_db: Any = Depends(get_mongo_db)
):
    """Server-Sent Events (SSE) streaming endpoint for live question generation."""
    return StreamingResponse(
        stream_question_generator(payload, mongo_db),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

@router.post("/questions/batch")
async def generate_question_batch(
    payload: GenerateQuestionBatchSchema,
    current_user: User = Depends(get_current_user),
    mongo_db: Any = Depends(get_mongo_db)
):
    """Retrieves or creates a mini-batch of 3 to 5 client-safe questions for topic & concept_id."""
    try:
        questions = await get_or_generate_question_batch(
            topic=payload.topic,
            concept_id=payload.concept_id,
            concept_type=payload.concept_type,
            difficulty=payload.difficulty,
            count=payload.count,
            mongo_db=mongo_db
        )
        return {
            "topic": payload.topic,
            "concept_id": payload.concept_id,
            "count": len(questions),
            "questions": questions
        }
    except Exception as e:
        logger.error(f"[generate_question_batch] Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Question batch generation failed: {str(e)}")

@router.post("/attempts/submit")
async def submit_attempt(
    payload: SubmitAttemptSchema,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    mongo_db: Any = Depends(get_mongo_db)
):
    """Evaluates user attempt, updates mastery, and auto-generates next question with increased difficulty and updated question count."""
    q_collection = mongo_db.get_collection("questions")
    question_doc = await q_collection.find_one({"_id": payload.question_id})

    if not question_doc:
        question_doc = await q_collection.find_one({"id": payload.question_id})

    if not question_doc:
        # Synthetic fallback. Code questions are inferred from execution
        # metrics so the all-test-cases gate still applies.
        is_code_hint = bool(payload.total_test_cases) or payload.test_cases_passed is not None
        question_doc = {
            "_id": payload.question_id,
            "id": payload.question_id,
            "topic": payload.topic,
            "concept_id": payload.concept_id,
            "concept_type": "code" if is_code_hint else "conceptual",
            "type": "practice" if is_code_hint else "mcq",
            "question": "Practice check",
            "correct_option": None if is_code_hint else "correct",
            "execution_method": "llm_rubric",
            "test_cases": [],
        }

    # 1. Multi-Signal Evaluation
    eval_result = await evaluate_user_attempt(
        question_doc=question_doc,
        user_answer=payload.answer,
        test_run_count=payload.test_run_count or 0,
        test_cases_passed=payload.test_cases_passed,
        total_test_cases=payload.total_test_cases
    )
    score = eval_result["score"]

    breakdown = eval_result["evaluation_breakdown"]
    is_passed = bool(breakdown.get("correct", False) or score >= 0.7)

    # Hard execution gate: a coding question is only passed when EVERY
    # generated test case passes. The LLM rubric can never override this.
    q_type = question_doc.get("type", "mcq")
    q_concept_type = question_doc.get("concept_type", "conceptual")
    is_code_question = q_type == "practice" or q_concept_type in ("code", "practice")
    total_tests = payload.total_test_cases
    if total_tests is None:
        total_tests = len(question_doc.get("test_cases") or [])
    passed_tests = payload.test_cases_passed or 0

    if is_code_question:
        if total_tests <= 0:
            is_passed = False
            score = min(score, 0.5)
            breakdown["correct"] = False
            breakdown["execution_gate"] = "No test cases available for this question; retry generation."
            breakdown["feedback"] = (
                "This coding question has no runnable test cases yet. "
                "Regenerate the question or retry so test cases are produced."
            )
        elif passed_tests < total_tests:
            is_passed = False
            score = min(score, 0.5)
            breakdown["correct"] = False
            breakdown["execution_gate"] = (
                f"{passed_tests}/{total_tests} test cases passed — all cases must pass."
            )
            breakdown["feedback"] = (
                f"Your solution passed {passed_tests}/{total_tests} test cases. "
                "Fix the failing cases and run the tests again before advancing."
            )

    # 2. Insert Postgres attempt record
    attempt_record = Attempt(
        user_id=current_user.id,
        question_id=payload.question_id,
        concept_id=payload.concept_id,
        answer=payload.answer,
        score=score,
        evaluation_breakdown=breakdown
    )
    db.add(attempt_record)
    await db.commit()

    # 3. Update concept mastery and check remediation triggers
    mastery_update = await update_user_mastery_and_check_remediation(
        user_id=current_user.id,
        topic=payload.topic,
        concept_id=payload.concept_id,
        attempt_score=score,
        difficulty=payload.difficulty,
        db=db,
        mongo_db=mongo_db
    )

    feedback_msg = breakdown.get("feedback") or ("Correct answer! Concept mastery updated." if is_passed else "Incorrect answer. Review concept guidelines and try again.")

    # 4. Progressive Question Count & Node Tracking in MongoDB learning_paths
    lp_collection = mongo_db.get_collection("learning_paths") if mongo_db is not None else None
    topic_slug = slugify_topic(payload.topic)
    lp_doc = None
    if lp_collection is not None:
        lp_doc = await lp_collection.find_one({"user_id": str(current_user.id), "topic": topic_slug})
        if not lp_doc:
            lp_doc = await lp_collection.find_one({"user_id": str(current_user.id), "topic": topic_slug.lower()})

    node_target_count = payload.target_question_count or 3
    node_questions_completed = 0
    node_correct_count = 0

    if lp_doc and "nodes" in lp_doc:
        nodes_list = lp_doc.get("nodes", [])
        updated_node = False
        for node in nodes_list:
            cid = node.get("concept_id") or node.get("id")
            if cid == payload.concept_id:
                node_target_count = node.get("target_question_count") or payload.target_question_count or 3
                node["target_question_count"] = node_target_count
                node["total_attempts"] = node.get("total_attempts", 0) + 1
                if is_passed:
                    node["correct_count"] = node.get("correct_count", 0) + 1
                    node["questions_completed"] = node["correct_count"]
                else:
                    node["questions_completed"] = node.get("correct_count", 0)

                node_questions_completed = node["questions_completed"]
                node_correct_count = node.get("correct_count", 0)

                if node_correct_count >= node_target_count or mastery_update.get("status") in ["mastered", "known"]:
                    node["status"] = "mastered"
                else:
                    if node.get("status") not in ["mastered", "known"]:
                        node["status"] = "in_progress"
                updated_node = True
                break

        if updated_node:
            await lp_collection.update_one({"_id": lp_doc["_id"]}, {"$set": {"nodes": nodes_list}})

    curr_count = payload.current_question_count
    target_count = node_target_count

    if is_passed:
        next_count = curr_count + 1
        next_difficulty = min(5, payload.difficulty + 1)
    else:
        next_count = curr_count
        next_difficulty = payload.difficulty

    # A concept is completed ONLY if the current attempt passed AND target correct count or EMA 0.85+ is reached
    is_concept_completed = False
    if is_passed:
        is_concept_completed = (
            next_count > target_count or
            node_correct_count >= target_count or
            mastery_update.get("new_mastery", 0.0) >= 0.85
        )

    if is_concept_completed:
        try:
            stmt_m = select(ConceptMastery).where(
                ConceptMastery.user_id == current_user.id,
                ConceptMastery.topic == slugify_topic(payload.topic),
                ConceptMastery.concept_id == payload.concept_id
            )
            res_m = await db.execute(stmt_m)
            rec_m = res_m.scalar_one_or_none()
            if rec_m:
                rec_m.status = "mastered"
                rec_m.mastery = max(rec_m.mastery, 0.85)
                await db.commit()

            await record_user_completed_node(
                user_id=current_user.id,
                topic_slug=slugify_topic(payload.topic),
                concept_id=payload.concept_id,
                mongo_db=mongo_db,
                db=db
            )
        except Exception as cm_err:
            logger.warning(f"Could not update ConceptMastery or user_topics status to mastered: {cm_err}")

    # 5. Sync learning path progress (unlock downstream nodes & auto-progress level if completed)
    lp_update = None
    try:
        lp_update = await sync_learning_path_progress(
            user_id=current_user.id,
            topic_slug=slugify_topic(payload.topic),
            mongo_db=mongo_db,
            db=db
        )
    except Exception as sync_err:
        logger.warning(f"Could not sync learning path progress: {sync_err}")

    # Exclude heavy 'nodes' array from response payload to keep payload lightweight
    if lp_update and isinstance(lp_update, dict) and "nodes" in lp_update:
        lp_update = {k: v for k, v in lp_update.items() if k != "nodes"}

    # The next question is streamed by the client after this response, which
    # keeps submission latency low. Only embed it when explicitly cached: the
    # client falls back to POST /questions/generate/stream otherwise.
    next_question = None
    if payload.include_next_question and is_passed and not is_concept_completed:
        try:
            next_question = await get_or_generate_question(
                topic=payload.topic,
                concept_id=payload.concept_id,
                concept_type=question_doc.get("concept_type", "conceptual"),
                difficulty=next_difficulty,
                language=payload.language or "python",
                explanation_language=payload.explanation_language or "english",
                mongo_db=mongo_db
            )
        except Exception as q_err:
            logger.warning(f"Could not generate next question for progressive flow: {q_err}")
            next_question = None

    return {
        "attempt_id": str(attempt_record.id),
        "score": score,
        "passed": is_passed,
        "feedback": feedback_msg,
        "evaluation_breakdown": breakdown,
        "mastery_update": mastery_update,
        "learning_path_update": lp_update,
        "current_question_count": curr_count,
        "next_question_count": next_count,
        "target_question_count": target_count,
        "questions_completed": node_questions_completed,
        "correct_count": node_correct_count,
        "next_difficulty": next_difficulty,
        "is_concept_completed": is_concept_completed,
        "next_question": next_question
    }
