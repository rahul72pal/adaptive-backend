"""Per-user LLM configuration storage and resolution."""

import uuid
from typing import Any, Dict, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import decrypt_secret, encrypt_secret
from app.db.models import UserLLMConfig
from app.utils.exceptions import DeepSeekAPIException

SUPPORTED_PROVIDERS = {"openai", "gemini", "anthropic", "openrouter", "custom", "openai-compatible"}


def _clean_optional(value: Optional[str]) -> Optional[str]:
    value = (value or "").strip()
    return value or None


async def get_user_llm_config(db: AsyncSession, user_id: uuid.UUID) -> Optional[UserLLMConfig]:
    stmt = select(UserLLMConfig).where(UserLLMConfig.user_id == user_id)
    return (await db.execute(stmt)).scalar_one_or_none()


async def upsert_user_llm_config(
    db: AsyncSession,
    user_id: uuid.UUID,
    *,
    provider: str,
    api_key: Optional[str],
    model: str,
    base_url: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 4096,
) -> UserLLMConfig:
    """Create or update a user's LLM config.

    ``api_key`` may be omitted on update to keep the previously stored key.
    """
    provider = (provider or "").lower().strip()
    if provider not in SUPPORTED_PROVIDERS:
        raise DeepSeekAPIException(
            message=f"Unsupported provider '{provider}'. Supported: {sorted(SUPPORTED_PROVIDERS)}",
            status_code=400,
        )

    model = (model or "").strip()
    if not model:
        raise DeepSeekAPIException(message="model is required.", status_code=400)

    base_url = _clean_optional(base_url)
    if provider in ("custom", "openai-compatible") and not base_url:
        raise DeepSeekAPIException(message="Custom provider requires base_url.", status_code=400)

    if not 0.0 <= float(temperature) <= 2.0:
        raise DeepSeekAPIException(message="temperature must be between 0.0 and 2.0.", status_code=400)
    if not 1 <= int(max_tokens) <= 128000:
        raise DeepSeekAPIException(message="max_tokens must be between 1 and 128000.", status_code=400)

    config = await get_user_llm_config(db, user_id)
    if config is None:
        if not (api_key or "").strip():
            raise DeepSeekAPIException(message="api_key is required for a new configuration.", status_code=400)
        config = UserLLMConfig(user_id=user_id)
        db.add(config)

    config.provider = provider
    config.model = model
    config.base_url = base_url
    config.temperature = float(temperature)
    config.max_tokens = int(max_tokens)
    if (api_key or "").strip():
        config.api_key_encrypted = encrypt_secret(api_key.strip())

    await db.flush()
    return config


async def delete_user_llm_config(db: AsyncSession, user_id: uuid.UUID) -> bool:
    config = await get_user_llm_config(db, user_id)
    if config is None:
        return False
    await db.delete(config)
    await db.flush()
    return True


def serialize_user_llm_config(config: UserLLMConfig, *, reveal_api_key: bool = False) -> Dict[str, Any]:
    """Public representation; never leaks the key unless explicitly requested."""
    payload: Dict[str, Any] = {
        "id": str(config.id),
        "provider": config.provider,
        "model": config.model,
        "base_url": config.base_url,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "has_api_key": bool(config.api_key_encrypted),
        "updated_at": config.updated_at.isoformat() if config.updated_at else None,
    }
    if reveal_api_key and config.api_key_encrypted:
        payload["api_key"] = decrypt_secret(config.api_key_encrypted)
    return payload


def config_to_llm_dict(config: UserLLMConfig) -> Dict[str, Any]:
    """Runtime dict consumed by the LLM gateway."""
    return {
        "provider": config.provider,
        "api_key": decrypt_secret(config.api_key_encrypted),
        "model": config.model,
        "base_url": config.base_url,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
    }
