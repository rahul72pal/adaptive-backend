"""Per-user LLM configuration API.

Endpoints let an authenticated user inspect, create, update and delete the LLM
provider credentials used for their own requests. When a saved configuration
exists it takes precedence over the server-wide environment settings.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.dependencies import get_current_user
from app.db.models import User
from app.db.postgres import get_db
from app.llm.config import LLMProvider
from app.services.user_llm_config_service import (
    delete_user_llm_config,
    get_user_llm_config,
    serialize_user_llm_config,
    upsert_user_llm_config,
)
from app.utils.exceptions import DeepSeekAPIException

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/user/llm-config", tags=["User LLM Configuration"])


class LLMConfigUpsertSchema(BaseModel):
    provider: str = Field(default="openai", description="openai | gemini | anthropic | openrouter | custom")
    api_key: Optional[str] = Field(default=None, description="Omit to keep the stored key on update")
    model: str = Field(..., min_length=1)
    base_url: Optional[str] = None
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=1, le=128000)


class LLMConfigResponse(BaseModel):
    configured: bool
    source: str = "user"
    config: Optional[dict] = None


def _handle(exc: DeepSeekAPIException) -> HTTPException:
    return HTTPException(status_code=exc.status_code, detail=exc.message)


@router.get("", response_model=LLMConfigResponse)
async def read_my_llm_config(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Return the current user's saved LLM configuration (API key redacted)."""
    config = await get_user_llm_config(db, current_user.id)
    if config is None:
        return LLMConfigResponse(configured=False, source="environment", config=None)
    return LLMConfigResponse(
        configured=True,
        source="user",
        config=serialize_user_llm_config(config, reveal_api_key=False),
    )


@router.put("", response_model=LLMConfigResponse)
async def save_my_llm_config(
    payload: LLMConfigUpsertSchema,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Create or update the current user's LLM configuration."""
    provider = (payload.provider or "").strip().lower()
    valid = {p.value for p in LLMProvider}
    if provider == "openai-compatible":
        provider = "custom"
    if provider not in valid:
        raise HTTPException(status_code=400, detail=f"Unsupported provider '{payload.provider}'.")

    try:
        config = await upsert_user_llm_config(
            db,
            current_user.id,
            provider=provider,
            api_key=payload.api_key,
            model=payload.model,
            base_url=payload.base_url,
            temperature=payload.temperature,
            max_tokens=payload.max_tokens,
        )
    except DeepSeekAPIException as exc:
        raise _handle(exc)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    logger.info("[UserLLMConfig] Saved provider='%s' for user=%s", config.provider, current_user.id)
    return LLMConfigResponse(
        configured=True,
        source="user",
        config=serialize_user_llm_config(config, reveal_api_key=False),
    )


@router.delete("")
async def delete_my_llm_config(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Remove the user's saved config, reverting to server environment settings."""
    deleted = await delete_user_llm_config(db, current_user.id)
    if not deleted:
        raise HTTPException(status_code=404, detail="No saved LLM configuration found.")
    logger.info("[UserLLMConfig] Deleted config for user=%s", current_user.id)
    return {"status": "deleted"}
