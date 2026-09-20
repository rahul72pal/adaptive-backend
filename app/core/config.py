import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # LLM Provider Configuration
    # Supports: openai, openrouter, gemini, groq, custom
    LLM_PROVIDER: str = "openai"
    
    # OpenAI & OpenRouter (OpenAI-compatible) API Specs
    OPENAI_API_KEY: str = ""
    OPENAI_API_BASE: str = "https://api.openai.com/v1"
    OPENROUTER_API_KEY: str = ""
    LLM_MODEL_NAME: str = "gpt-4o-mini"
    
    # Gemini API Specs
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL_NAME: str = "gemini-1.5-flash"
    
    # Groq API Specs
    GROQ_API_KEY: str = ""
    GROQ_MODEL_NAME: str = "llama-3.3-70b-versatile"

    # Custom / OpenAI-Compatible (vLLM, Ollama, LocalAI, NVIDIA NIM) Specs
    CUSTOM_API_KEY: str = ""
    CUSTOM_API_BASE: str = ""
    CUSTOM_BASE_URL: str = ""
    CUSTOM_MODEL_NAME: str = ""
    
    # LLM Generation Limits
    LLM_MAX_TOKENS: int = 5000

    # Databases
    POSTGRES_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/adaptive_learn"
    SQLITE_FALLBACK_URL: str = "sqlite+aiosqlite:///./adaptive_learn_fallback.db"
    MONGO_URL: str = "mongodb://localhost:27017"
    MONGO_DB_NAME: str = "adaptive_learn"
    REDIS_URL: str = "redis://localhost:6379/0"

    # Security & Auth
    JWT_SECRET: str = "super-secret-jwt-key-change-in-production-123456789"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440

    # Application
    ENV: str = "development"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

settings = Settings()
