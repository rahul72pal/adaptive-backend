import logging
from typing import Optional, Any
import redis.asyncio as redis
from app.core.config import settings

logger = logging.getLogger(__name__)

class InMemoryRedisFallback:
    """In-memory Redis substitute when live Redis server is unavailable."""
    def __init__(self):
        self._store = {}

    async def get(self, key: str) -> Optional[str]:
        return self._store.get(key)

    async def set(self, key: str, value: str, ex: Optional[int] = None) -> bool:
        self._store[key] = str(value)
        return True

    async def delete(self, key: str) -> int:
        if key in self._store:
            del self._store[key]
            return 1
        return 0

    async def incr(self, key: str) -> int:
        val = int(self._store.get(key, 0)) + 1
        self._store[key] = str(val)
        return val

redis_client: Optional[Any] = None
redis_checkpointer: Optional[Any] = None

async def init_redis() -> None:
    """Initializes Redis client connection or fallback."""
    global redis_client
    try:
        r = redis.from_url(settings.REDIS_URL, decode_responses=True)
        await r.ping()
        redis_client = r
        logger.info("Successfully connected to Redis.")
    except Exception as e:
        logger.warning(f"Redis connection failed ({e}). Activating in-memory Redis fallback...")
        redis_client = InMemoryRedisFallback()

def get_redis_client():
    """Dependency for providing Redis client handle."""
    global redis_client
    if redis_client is None:
        redis_client = InMemoryRedisFallback()
    return redis_client

def get_redis_checkpointer():
    """Dependency for providing LangGraph state checkpointer handle."""
    global redis_checkpointer
    if redis_checkpointer is None:
        from langgraph.checkpoint.memory import MemorySaver
        redis_checkpointer = MemorySaver()
    return redis_checkpointer
