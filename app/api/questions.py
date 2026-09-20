import uuid
from typing import Optional, Any, List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field

from app.db.postgres import get_db
from app.db.mongo import get_mongo_db
from app.core.dependencies import get_current_user
from app.db.models import User, Attempt
from app.engine.question_engine import get_or_generate_question, get_or_generate_question_batch
from app.engine.evaluation_engine import evaluate_user_attempt
from app.engine.mastery_engine import update_user_mastery_and_check_remediation
from app.engine.curriculum_engine import sync_learning_path_progress, slugify_topic

router = APIRouter(prefix="/api", tags=["Question Engine & Evaluation"])

class GenerateQuestionSchema(BaseModel):
    topic: str
    concept_id: str
    concept_type: str = "conceptual"
    difficulty: int = 3

class GenerateQuestionBatchSchema(BaseModel):
    topic: str
    concept_id: str
    concept_type: str = "conceptual"
    difficulty: int = 3
    count: int = Field(default=3, ge=1, le=10)

class SubmitAttemptSchema(BaseModel):
    question_id: str
    concept_id: str
    topic: str
    difficulty: int = 1
    answer: str
    current_question_count: int = Field(default=1, ge=1)
    target_question_count: Optional[int] = Field(default=10, ge=1)

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
            mongo_db=mongo_db
        )
        return question_doc
    except Exception as e:
        logger.error(f"[generate_question] Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Question generation failed: {str(e)}")

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
    eval_result = await evaluate_user_attempt(question_doc, payload.answer)
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

    # 4. Sync learning path progress (unlock downstream nodes & auto-progress level if completed)
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

    # 5. Progressive Question Count & Difficulty Level Tracking
    curr_count = payload.current_question_count
    target_count = payload.target_question_count or 10

    if is_passed:
        next_count = curr_count + 1
        next_difficulty = min(5, payload.difficulty + 1)
    else:
        next_count = curr_count
        next_difficulty = payload.difficulty

    is_concept_completed = next_count > target_count

    next_question = None
    if not is_concept_completed:
        try:
            next_question = await get_or_generate_question(
                topic=payload.topic,
                concept_id=payload.concept_id,
                concept_type=question_doc.get("concept_type", "conceptual"),
                difficulty=next_difficulty,
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
        "next_difficulty": next_difficulty,
        "is_concept_completed": is_concept_completed,
        "next_question": next_question
    }
