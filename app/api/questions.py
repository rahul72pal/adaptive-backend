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

@router.post("/questions/generate")
async def generate_question(
    payload: GenerateQuestionSchema,
    current_user: User = Depends(get_current_user),
    mongo_db: Any = Depends(get_mongo_db)
):
    """Retrieves a single client-safe question for topic & concept_id from MongoDB cache or triggers mini-batch LLM generation."""
    question_doc = await get_or_generate_question(
        topic=payload.topic,
        concept_id=payload.concept_id,
        concept_type=payload.concept_type,
        difficulty=payload.difficulty,
        mongo_db=mongo_db
    )
    return question_doc

@router.post("/questions/batch")
async def generate_question_batch(
    payload: GenerateQuestionBatchSchema,
    current_user: User = Depends(get_current_user),
    mongo_db: Any = Depends(get_mongo_db)
):
    """Retrieves or creates a mini-batch of 3 to 5 client-safe questions for topic & concept_id."""
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

@router.post("/attempts/submit")
async def submit_attempt(
    payload: SubmitAttemptSchema,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    mongo_db: Any = Depends(get_mongo_db)
):
    """Evaluates user attempt (Deterministic + LLM Rubric), updates mastery, and checks remediation triggers."""
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

    return {
        "attempt_id": str(attempt_record.id),
        "score": score,
        "passed": is_passed,
        "feedback": feedback_msg,
        "evaluation_breakdown": breakdown,
        "mastery_update": mastery_update
    }
