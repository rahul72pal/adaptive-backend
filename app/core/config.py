import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables / .env file.

    LLM provider selection
    ----------------------
    Set ``LLM_PROVIDER`` to one of: openai | gemini | anthropic | custom

    Provider-specific vars
    ----------------------
    OpenAI:     OPENAI_API_KEY, LLM_MODEL_NAME, OPENAI_API_BASE (optional)
    Gemini:     GEMINI_API_KEY, GEMINI_MODEL_NAME
    Anthropic:  ANTHROPIC_API_KEY, ANTHROPIC_MODEL_NAME
    Custom:     CUSTOM_API_KEY, CUSTOM_BASE_URL (or CUSTOM_API_BASE), CUSTOM_MODEL_NAME

    Shared generation limits
    ------------------------
    LLM_MAX_TOKENS   — hard cap on output tokens for all providers
    LLM_TEMPERATURE  — sampling temperature (0.0 – 1.0 for most providers)
    """

    # ------------------------------------------------------------------
    # Provider selection
    # ------------------------------------------------------------------
    LLM_PROVIDER: str = "openai"  # openai | gemini | anthropic | custom

    # ------------------------------------------------------------------
    # OpenAI
    # ------------------------------------------------------------------
    OPENAI_API_KEY: str = ""
    OPENAI_API_BASE: str = "https://api.openai.com/v1"

    # ------------------------------------------------------------------
    # Gemini (Google AI Studio)
    # ------------------------------------------------------------------
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL_NAME: str = "gemini-1.5-flash"

    # ------------------------------------------------------------------
    # Anthropic
    # ------------------------------------------------------------------
    ANTHROPIC_API_KEY: str = ""
    ANTHROPIC_MODEL_NAME: str = "claude-3-5-sonnet-20241022"

    # ------------------------------------------------------------------
    # OpenRouter
    # ------------------------------------------------------------------
    OPENROUTER_API_KEY: str = ""
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    OPENROUTER_MODEL_NAME: str = ""

    # ------------------------------------------------------------------
    # Custom / OpenAI-compatible (vLLM, Ollama, NVIDIA NIM, LM Studio …)
    # ------------------------------------------------------------------
    CUSTOM_API_KEY: str = ""
    CUSTOM_BASE_URL: str = ""
    CUSTOM_API_BASE: str = ""   # alias for CUSTOM_BASE_URL
    CUSTOM_MODEL_NAME: str = ""

    # ------------------------------------------------------------------
    # Shared model / generation limits
    # ------------------------------------------------------------------
    LLM_MODEL_NAME: str = "gpt-4o-mini"   # used when provider-specific var is absent
    LLM_MAX_TOKENS: int = 4096
    LLM_TEMPERATURE: float = 0.7

    # ------------------------------------------------------------------
    # Databases
    # ------------------------------------------------------------------
    POSTGRES_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/adaptive_learn"
    SQLITE_FALLBACK_URL: str = "sqlite+aiosqlite:///./adaptive_learn_fallback.db"
    MONGO_URL: str = "mongodb://localhost:27017"
    MONGO_DB_NAME: str = "adaptive_learn"
    REDIS_URL: str = "redis://localhost:6379/0"

    # ------------------------------------------------------------------
    # Security & Auth
    # ------------------------------------------------------------------
    JWT_SECRET: str = "super-secret-jwt-key-change-in-production-123456789"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440

    # ------------------------------------------------------------------
    # Code execution sandbox (OneCompiler hosted runner)
    # ------------------------------------------------------------------
    CODE_EXECUTION_ENABLED: bool = True
    # OneCompiler code-execution API. The key is read from ONE_COMPILER_API.
    ONE_COMPILER_API: str = ""
    ONE_COMPILER_API_URL: str = "https://api.onecompiler.com/api/code/exec"
    ONE_COMPILER_TIMEOUT_SECONDS: float = 25.0

    # ------------------------------------------------------------------
    # Application
    # ------------------------------------------------------------------
    ENV: str = "development"
    APP_URL: str = "http://localhost:8000"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()
