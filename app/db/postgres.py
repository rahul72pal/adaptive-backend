import logging
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from app.core.config import settings
from app.db.models import Base

logger = logging.getLogger(__name__)

engine = None
AsyncSessionLocal = None

async def init_db() -> None:
    """Initializes Database engine & tables. Uses SQLite fallback if PostgreSQL is unavailable."""
    global engine, AsyncSessionLocal

    postgres_url = settings.POSTGRES_URL
    if postgres_url.startswith("postgresql://"):
        postgres_url = postgres_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    elif postgres_url.startswith("postgres://"):
        postgres_url = postgres_url.replace("postgres://", "postgresql+asyncpg://", 1)

    connect_args = {
        "timeout": 15,
        "command_timeout": 10
    }
    if "render.com" in postgres_url or "sslmode=require" in postgres_url or "ssl=require" in postgres_url:
        connect_args["ssl"] = "require"

    try:
        pg_engine = create_async_engine(
            postgres_url,
            echo=False,
            pool_pre_ping=False,
            pool_recycle=60,
            pool_timeout=10,
            connect_args=connect_args
        )
        async with pg_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        engine = pg_engine
        AsyncSessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        logger.info("Successfully initialized PostgreSQL database.")
    except Exception as e:
        logger.warning(f"PostgreSQL initialization failed ({e}). Falling back to SQLite database...")
        sqlite_url = settings.SQLITE_FALLBACK_URL
        sqlite_engine = create_async_engine(sqlite_url, echo=False)
        async with sqlite_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        engine = sqlite_engine
        AsyncSessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        logger.info("Successfully initialized SQLite fallback database.")

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for providing asynchronous SQLAlchemy sessions."""
    global AsyncSessionLocal
    if AsyncSessionLocal is None:
        await init_db()

    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
