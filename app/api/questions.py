import uuid
from typing import Optional, Any, List
from fastapi import APIRouter, Depends, HTTPException, status
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


import json
import logging
import random
from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)

async def stream_question_generator(payload: GenerateQuestionSchema, mongo_db: Any):
    from app.engine.question_engine import get_or_generate_question, sanitize_question_for_client, DIFFICULTY_DESCRIPTIONS
    from app.llm.gateway import call_llm_gateway_stream, parse_json_from_llm_response
    from app.llm.prompts import QUESTION_GEN_PROMPT
    from app.llm.schemas import QuestionBatchSchema

    topic_clean = payload.topic.lower().strip()
    concept_id = payload.concept_id.strip()
    concept_type = payload.concept_type.strip()
    difficulty = payload.difficulty

    # 1. Check MongoDB cache first for immediate cached response
    if mongo_db is not None:
        try:
            q_col = mongo_db.get_collection("questions")
            query = {
                "topic": topic_clean,
                "concept_id": concept_id,
                "difficulty": difficulty
            }
            if concept_type:
                query["concept_type"] = concept_type

            cached_docs = await q_col.find(query).to_list(10)
            if cached_docs:
                doc = random.choice(cached_docs)
                safe_q = sanitize_question_for_client(doc)
                yield f"data: {json.dumps({'event': 'cached', 'question': safe_q, 'markdown': safe_q.get('markdown', '')})}\n\n"
                return
        except Exception as e:
            logger.warning(f"[stream_question_generator] Cache check warning: {e}")

    # 2. Stream tokens live from LLM
    yield f"data: {json.dumps({'event': 'start', 'status': 'generating'})}\n\n"
    diff_desc = DIFFICULTY_DESCRIPTIONS.get(difficulty, "standard conceptual understanding")
    full_tokens = []

    try:
        async for chunk_text in call_llm_gateway_stream(
            QUESTION_GEN_PROMPT,
            {
                "topic": topic_clean,
                "concept_id": concept_id,
                "pattern": "none",
                "concept_type": concept_type,
                "difficulty": difficulty,
                "difficulty_description": diff_desc,
                "question_type": "practice" if concept_type == "code" else "mcq",
                "count": 1,
                "previous_question_titles": "None"
            },
            QuestionBatchSchema
        ):
            full_tokens.append(chunk_text)
            yield f"data: {json.dumps({'event': 'token', 'delta': chunk_text})}\n\n"

        raw_json_str = "".join(full_tokens)
        parsed = parse_json_from_llm_response(raw_json_str, QuestionBatchSchema, topic_hint=concept_id)
        
        raw_questions = getattr(parsed, "questions", [])
        if not raw_questions:
            fallback_q = await get_or_generate_question(topic_clean, concept_id, concept_type, difficulty, mongo_db)
            yield f"data: {json.dumps({'event': 'complete', 'question': fallback_q, 'markdown': fallback_q.get('markdown', '')})}\n\n"
            return

        q_item = raw_questions[0]
        q_dict = q_item.model_dump() if hasattr(q_item, "model_dump") else dict(q_item)

        q_id = f"q_{uuid.uuid4().hex[:8]}"
        q_doc = {
            "_id": q_id,
            "id": q_id,
            "topic": topic_clean,
            "concept_id": concept_id,
            "concept_type": concept_type,
            "difficulty": difficulty,
            **q_dict
        }

        if mongo_db is not None:
            try:
                q_col = mongo_db.get_collection("questions")
                await q_col.insert_one(q_doc)
            except Exception:
                pass

        safe_q = sanitize_question_for_client(q_doc)
        yield f"data: {json.dumps({'event': 'complete', 'question': safe_q, 'markdown': safe_q.get('markdown', '')})}\n\n"
    except Exception as err:
        logger.error(f"[stream_question_generator] Error: {err}", exc_info=True)
        try:
            fallback_q = await get_or_generate_question(topic_clean, concept_id, concept_type, difficulty, mongo_db)
            yield f"data: {json.dumps({'event': 'complete', 'question': fallback_q, 'markdown': fallback_q.get('markdown', '')})}\n\n"
        except Exception:
            yield f"data: {json.dumps({'event': 'error', 'message': str(err)})}\n\n"


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
        # Fallback question doc if ID is synthetic
        question_doc = {
            "_id": payload.question_id,
            "topic": payload.topic,
            "concept_id": payload.concept_id,
            "concept_type": "conceptual",
            "type": "mcq",
            "question": "Practice check",
            "correct_option": "correct",
            "execution_method": "llm_rubric"
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

    next_question = None
    if is_passed and not is_concept_completed:
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
