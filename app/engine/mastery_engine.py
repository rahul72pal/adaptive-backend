import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.db.models import ConceptMastery, Attempt

def calculate_ema_mastery(prev_mastery: float, attempt_score: float, difficulty: int = 1) -> float:
    """Calculates Exponential Moving Average (EMA) mastery update with IRT difficulty scaling."""
    difficulty_weight = 0.8 + (difficulty * 0.08)  # Ranges from 0.88 to 1.2
    normalized_score = min(1.0, attempt_score * difficulty_weight)
    new_mastery = (0.6 * normalized_score) + (0.4 * prev_mastery)
    return round(new_mastery, 4)

async def update_user_mastery_and_check_remediation(
    user_id: uuid.UUID,
    topic: str,
    concept_id: str,
    attempt_score: float,
    difficulty: int,
    db: AsyncSession,
    mongo_db: Any
) -> Dict[str, Any]:
    """Updates concept mastery in PostgreSQL and checks whether adaptive remediation or level unlock is needed."""
    stmt = select(ConceptMastery).where(
        ConceptMastery.user_id == user_id,
        ConceptMastery.topic == topic,
        ConceptMastery.concept_id == concept_id
    )
    res = await db.execute(stmt)
    mastery_record = res.scalar_one_or_none()

    if not mastery_record:
        mastery_record = ConceptMastery(
            user_id=user_id,
            topic=topic,
            concept_id=concept_id,
            mastery=0.0,
            attempts=0,
            failures=0,
            status="learning"
        )
        db.add(mastery_record)

    prev_mastery = mastery_record.mastery
    new_mastery = calculate_ema_mastery(prev_mastery, attempt_score, difficulty)

    mastery_record.mastery = new_mastery
    mastery_record.attempts += 1
    mastery_record.last_updated = datetime.now(timezone.utc)

    is_pass = attempt_score >= 0.70
    if is_pass:
        if new_mastery >= 0.85:
            mastery_record.status = "mastered"
    else:
        mastery_record.failures += 1

    await db.commit()

    # Check for consecutive failures remediation pattern
    remediation_required = False
    remediation_action = "continue"
    
    if mastery_record.failures >= 3:
        remediation_required = True
        remediation_action = "spot_check_prereq"

    # Check rolling 10-attempt failure rate across topic
    stmt_attempts = select(Attempt).where(Attempt.user_id == user_id).order_by(Attempt.created_at.desc()).limit(10)
    att_res = await db.execute(stmt_attempts)
    recent_attempts = att_res.scalars().all()

    if len(recent_attempts) >= 5:
        failed_count = sum(1 for a in recent_attempts if a.score < 0.70)
        failure_rate = failed_count / len(recent_attempts)
        if failure_rate >= 0.60:
            remediation_required = True
            remediation_action = "re_diagnose"

    return {
        "concept_id": concept_id,
        "prev_mastery": prev_mastery,
        "new_mastery": new_mastery,
        "status": mastery_record.status,
        "attempts": mastery_record.attempts,
        "failures": mastery_record.failures,
        "remediation_required": remediation_required,
        "remediation_action": remediation_action
    }

async def calculate_topic_progress(
    user_id: uuid.UUID,
    topic: str,
    db: AsyncSession,
    mongo_db: Any
) -> Dict[str, Any]:
    """Calculates overall progress breakdown and current unlocked level threshold."""
    stmt = select(ConceptMastery).where(
        ConceptMastery.user_id == user_id,
        ConceptMastery.topic == topic
    )
    res = await db.execute(stmt)
    records = res.scalars().all()

    if not records:
        return {
            "topic": topic,
            "overall_mastery": 0.0,
            "current_level": "beginner",
            "concept_breakdown": []
        }

    total_concepts = len(records)
    avg_mastery = sum(r.mastery for r in records) / total_concepts
    mastered_count = sum(1 for r in records if r.status in ["known", "mastered"])
    coverage = (mastered_count / total_concepts) * 100.0

    # Level unlock threshold verification
    if coverage >= 90.0 and avg_mastery >= 0.90:
        current_level = "expert"
    elif coverage >= 90.0 and avg_mastery >= 0.85:
        current_level = "advanced"
    elif coverage >= 90.0 and avg_mastery >= 0.80:
        current_level = "intermediate"
    else:
        current_level = "beginner"

    breakdown = [
        {
            "concept_id": r.concept_id,
            "mastery": r.mastery,
            "status": r.status,
            "attempts": r.attempts,
            "failures": r.failures
        }
        for r in records
    ]

    return {
        "topic": topic,
        "overall_mastery": round(avg_mastery, 4),
        "coverage_percentage": round(coverage, 2),
        "current_level": current_level,
        "mastered_concepts_count": mastered_count,
        "total_concepts_count": total_concepts,
        "concept_breakdown": breakdown
    }
