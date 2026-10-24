import logging
from typing import AsyncGenerator
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import uuid
import logging

from app.db.postgres import get_db
from app.db.mongo import get_mongo_db
from app.db.redis_client import get_redis_client
from app.core.security import decode_access_token
from app.db.models import User
from app.llm.user_context import load_user_llm_config, set_active_llm_config, set_active_user_id

logger = logging.getLogger(__name__)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db)
) -> User:
    """Retrieves current authenticated User from JWT token."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    payload = decode_access_token(token)
    if not payload:
        raise credentials_exception
    
    user_id_str: str = payload.get("sub")
    if not user_id_str:
        raise credentials_exception
    
    try:
        user_uuid = uuid.UUID(user_id_str)
    except ValueError:
        raise credentials_exception

    stmt = select(User).where(User.id == user_uuid)
    result = await db.execute(stmt)
    user = result.scalar_one_or_none()
    
    if user is None:
        raise credentials_exception

    # Populate the request-scoped LLM config so the gateway uses this user's
    # credentials instead of the server-wide environment defaults. The lookup runs
    # inside a SAVEPOINT so a failure here (a database that predates the
    # user_llm_config table, or a key encrypted under an older JWT_SECRET) rolls
    # back only the savepoint instead of poisoning the request transaction,
    # which would make every later query in this request fail.
    set_active_user_id(user.id)
    try:
        async with db.begin_nested():
            config = await load_user_llm_config(db, user.id)
        set_active_llm_config(config)
    except Exception as exc:
        logger.warning("Failed to load LLM config for user %s: %s", user.id, exc)
        set_active_llm_config(None)

    return user
