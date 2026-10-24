import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.db.postgres import init_db
from app.db.mongo import init_mongo
from app.db.redis_client import init_redis

from app.api.auth import router as auth_router
from app.api.diagnostic import router as diagnostic_router
from app.api.learning import router as learning_router
from app.api.questions import router as questions_router
from app.api.progress import router as progress_router
from app.api.topics import router as topics_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("adaptive_backend")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager handling database connection setups on startup."""
    logger.info("Initializing databases and external clients...")
    await init_db()
    await init_mongo()
    await init_redis()
    try:
        from app.services.topic_catalog_service import get_catalog_service
        from app.db.postgres import AsyncSessionLocal
        from app.db.mongo import get_mongo_db
        cat_svc = get_catalog_service()
        if AsyncSessionLocal is not None:
            async with AsyncSessionLocal() as session:
                await cat_svc.sync_to_database(db=session, mongo_db=get_mongo_db())
    except Exception as sync_err:
        logger.warning(f"Catalog sync on startup: {sync_err}")
    logger.info("Adaptive Platform Backend startup complete.")
    yield
    logger.info("Adaptive Platform Backend shutting down...")

app = FastAPI(
    title="Adaptive Learning Platform API",
    description="Stateful LangGraph & Stateless LangChain Orchestrated Adaptive Backend",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Router Modules
app.include_router(auth_router)
app.include_router(diagnostic_router)
app.include_router(learning_router)
app.include_router(questions_router)
app.include_router(progress_router)
app.include_router(topics_router)

@app.get("/health", tags=["Health Check"])
async def health_check():
    """Health check endpoint for checking backend server status."""
    return {
        "status": "healthy",
        "env": settings.ENV,
        "service": "adaptive-backend"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
