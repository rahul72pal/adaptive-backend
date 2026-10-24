"""Request-scoped storage for a user's LLM configuration.

The middleware in ``main.py`` resolves the logged-in user's saved config and
stores the runtime dict here. Gateway functions read it so every LLM call made
during that request uses the user's provider/model instead of the env defaults.
"""

import uuid
from contextvars import ContextVar
from typing import Any, Dict, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.services.user_llm_config_service import config_to_llm_dict, get_user_llm_config

_active_user_llm_config: ContextVar[Optional[Dict[str, Any]]] = ContextVar(
    "active_user_llm_config", default=None
)
_active_user_id: ContextVar[Optional[uuid.UUID]] = ContextVar("active_user_id", default=None)


def set_active_llm_config(config: Optional[Dict[str, Any]]) -> None:
    _active_user_llm_config.set(config)


def get_active_llm_config() -> Optional[Dict[str, Any]]:
    return _active_user_llm_config.get()


def set_active_user_id(user_id: Optional[uuid.UUID]) -> None:
    _active_user_id.set(user_id)


def get_active_user_id() -> Optional[uuid.UUID]:
    return _active_user_id.get()


async def load_user_llm_config(db: AsyncSession, user_id: uuid.UUID) -> Optional[Dict[str, Any]]:
    """Fetch and decrypt the user's saved config, or ``None`` when absent."""
    row = await get_user_llm_config(db, user_id)
    if row is None:
        return None
    return config_to_llm_dict(row)
