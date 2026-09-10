import uuid
from typing import Any
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.postgres import get_db
from app.db.mongo import get_mongo_db
from app.core.dependencies import get_current_user
from app.db.models import User
from app.engine.mastery_engine import calculate_topic_progress

router = APIRouter(prefix="/api/progress", tags=["Progress & Mastery Engine"])

@router.get("/{user_id}/{topic}")
async def get_progress(
    user_id: str,
    topic: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    mongo_db: Any = Depends(get_mongo_db)
):
    """Retrieves deterministic progress summary, concept mastery breakdown, and current unlocked level."""
    try:
        user_uuid = uuid.UUID(user_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid user_id format.")

    progress = await calculate_topic_progress(
        user_id=user_uuid,
        topic=topic.lower(),
        db=db,
        mongo_db=mongo_db
    )
    return progress
