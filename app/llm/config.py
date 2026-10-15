"""
LLM Provider Configuration — Production-Level Pydantic Models
=============================================================

Defines strict, validated configuration contracts for every supported
LLM provider. All public entry-points validate at the boundary; internal
code trusts these typed objects.

Supported providers
-------------------
- openai        → OpenAI API (https://api.openai.com/v1)
- gemini        → Google Gemini (langchain_google_genai)
- anthropic     → Anthropic Claude (langchain_anthropic)
- custom        → Any OpenAI-compatible endpoint (vLLM, Ollama, NVIDIA NIM …)
                  Requires base_url.
"""

from __future__ import annotations

import os
from enum import Enum
from typing import Literal, Optional, Union, Annotated

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class LLMProvider(str, Enum):
    OPENAI = "openai"
    GEMINI = "gemini"
    ANTHROPIC = "anthropic"
    OPENROUTER = "openrouter"
    CUSTOM = "custom"


# ---------------------------------------------------------------------------
# Per-provider Config Models (discriminated union)
# ---------------------------------------------------------------------------

class OpenAIConfig(BaseModel):
    """Configuration for the OpenAI provider."""
    provider: Literal[LLMProvider.OPENAI] = LLMProvider.OPENAI
    api_key: str = Field(..., min_length=1, description="OpenAI API key")
    model: str = Field(..., min_length=1, description="e.g. gpt-4o, gpt-4o-mini")
    base_url: str = Field(
        default="https://api.openai.com/v1",
        description="Override for Azure OpenAI or proxy endpoints"
    )
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=1, le=128000)

    @field_validator("api_key")
    @classmethod
    def _reject_placeholder(cls, v: str) -> str:
        _check_placeholder(v, "openai")
        return v.strip()

    @field_validator("model")
    @classmethod
    def _strip_model(cls, v: str) -> str:
        return v.strip()


class GeminiConfig(BaseModel):
    """Configuration for Google Gemini provider."""
    provider: Literal[LLMProvider.GEMINI] = LLMProvider.GEMINI
    api_key: str = Field(..., min_length=1, description="Google AI Studio API key")
    model: str = Field(..., min_length=1, description="e.g. gemini-1.5-flash, gemini-2.0-flash")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=1, le=65536)

    @field_validator("api_key")
    @classmethod
    def _reject_placeholder(cls, v: str) -> str:
        _check_placeholder(v, "gemini")
        return v.strip()

    @field_validator("model")
    @classmethod
    def _strip_model(cls, v: str) -> str:
        return v.strip()


class AnthropicConfig(BaseModel):
    """Configuration for Anthropic Claude provider."""
    provider: Literal[LLMProvider.ANTHROPIC] = LLMProvider.ANTHROPIC
    api_key: str = Field(..., min_length=1, description="Anthropic API key")
    model: str = Field(..., min_length=1, description="e.g. claude-3-5-sonnet-20241022")
    temperature: float = Field(default=0.7, ge=0.0, le=1.0)
    max_tokens: int = Field(default=4096, ge=1, le=200000)

    @field_validator("api_key")
    @classmethod
    def _reject_placeholder(cls, v: str) -> str:
        _check_placeholder(v, "anthropic")
        return v.strip()

    @field_validator("model")
    @classmethod
    def _strip_model(cls, v: str) -> str:
        return v.strip()


class OpenRouterConfig(BaseModel):
    """Configuration for OpenRouter provider (https://openrouter.ai)."""
    provider: Literal[LLMProvider.OPENROUTER] = LLMProvider.OPENROUTER
    api_key: str = Field(..., min_length=1, description="OpenRouter API key (sk-or-v1-...)")
    model: str = Field(..., min_length=1, description="e.g. google/gemini-2.0-flash-001, deepseek/deepseek-chat")
    base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        description="OpenRouter API base URL"
    )
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=1, le=128000)

    @field_validator("api_key")
    @classmethod
    def _reject_placeholder(cls, v: str) -> str:
        _check_placeholder(v, "openrouter")
        return v.strip()

    @field_validator("model")
    @classmethod
    def _strip_model(cls, v: str) -> str:
        return v.strip()


class CustomConfig(BaseModel):
    """
    Configuration for any OpenAI-compatible endpoint
    (vLLM, Ollama, LocalAI, NVIDIA NIM, LM Studio, Azure …).

    ``base_url`` is mandatory for this provider type.
    """
    provider: Literal[LLMProvider.CUSTOM] = LLMProvider.CUSTOM
    api_key: str = Field(
        default="dummy-key",
        description="API key; use 'dummy-key' if the endpoint does not require auth"
    )
    model: str = Field(..., min_length=1, description="Model name as expected by the endpoint")
    base_url: str = Field(..., min_length=1, description="e.g. https://integrate.api.nvidia.com/v1")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=1, le=128000)

    @field_validator("base_url")
    @classmethod
    def _validate_url(cls, v: str) -> str:
        v = v.strip().rstrip("/")
        if not v.startswith(("http://", "https://")):
            raise ValueError(
                f"Custom LLM base_url must start with http:// or https://, got: '{v}'"
            )
        return v

    @field_validator("model")
    @classmethod
    def _strip_model(cls, v: str) -> str:
        return v.strip()


# ---------------------------------------------------------------------------
# Discriminated union — the single public type callers use
# ---------------------------------------------------------------------------

LLMConfig = Annotated[
    Union[OpenAIConfig, GeminiConfig, AnthropicConfig, OpenRouterConfig, CustomConfig],
    Field(discriminator="provider"),
]


# ---------------------------------------------------------------------------
# Factory — build from environment / settings object
# ---------------------------------------------------------------------------

def llm_config_from_settings(settings_obj: object) -> Union[OpenAIConfig, GeminiConfig, AnthropicConfig, OpenRouterConfig, CustomConfig]:
    """
    Construct a validated LLM config from the application ``Settings``
    object (pydantic-settings). Raises ``ValueError`` with a clear message
    if the environment is mis-configured.

    This is the **only** place that reads raw env-var strings.
    All other code receives a typed config object.
    """
    provider_raw: str = (
        getattr(settings_obj, "LLM_PROVIDER", "") or os.getenv("LLM_PROVIDER", "openai")
    ).strip().lower()

    try:
        provider = LLMProvider(provider_raw)
    except ValueError:
        supported = ", ".join(p.value for p in LLMProvider)
        raise ValueError(
            f"Unsupported LLM_PROVIDER='{provider_raw}'. "
            f"Supported: {supported}"
        )

    def _attr(name: str, fallback: str = "") -> str:
        val = getattr(settings_obj, name, None)
        if val is None or val == "":
            val = os.getenv(name, "") or fallback
        return str(val).strip()

    def _int_attr(name: str, fallback: int) -> int:
        try:
            return int(_attr(name, str(fallback)))
        except (ValueError, TypeError):
            return fallback

    def _float_attr(name: str, fallback: float) -> float:
        try:
            return float(_attr(name, str(fallback)))
        except (ValueError, TypeError):
            return fallback

    max_tokens = _int_attr("LLM_MAX_TOKENS", 4096)
    temperature = _float_attr("LLM_TEMPERATURE", 0.7)

    if provider == LLMProvider.OPENAI:
        return OpenAIConfig(
            api_key=_attr("OPENAI_API_KEY"),
            model=_attr("LLM_MODEL_NAME", "gpt-4o-mini"),
            base_url=_attr("OPENAI_API_BASE", "https://api.openai.com/v1"),
            temperature=temperature,
            max_tokens=max_tokens,
        )

    if provider == LLMProvider.GEMINI:
        return GeminiConfig(
            api_key=_attr("GEMINI_API_KEY"),
            model=_attr("GEMINI_MODEL_NAME") or _attr("LLM_MODEL_NAME", "gemini-1.5-flash"),
            temperature=temperature,
            max_tokens=max_tokens,
        )

    if provider == LLMProvider.ANTHROPIC:
        return AnthropicConfig(
            api_key=_attr("ANTHROPIC_API_KEY"),
            model=_attr("ANTHROPIC_MODEL_NAME") or _attr("LLM_MODEL_NAME", "claude-3-5-sonnet-20241022"),
            temperature=temperature,
            max_tokens=max_tokens,
        )

    if provider == LLMProvider.OPENROUTER:
        return OpenRouterConfig(
            api_key=_attr("OPENROUTER_API_KEY") or _attr("OPENAI_API_KEY"),
            model=_attr("OPENROUTER_MODEL_NAME") or _attr("LLM_MODEL_NAME", "google/gemini-2.0-flash-001"),
            base_url=_attr("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            temperature=temperature,
            max_tokens=max_tokens,
        )

    if provider == LLMProvider.CUSTOM:
        base_url = _attr("CUSTOM_BASE_URL") or _attr("CUSTOM_API_BASE")
        if not base_url:
            raise ValueError(
                "LLM_PROVIDER=custom requires CUSTOM_BASE_URL (or CUSTOM_API_BASE) "
                "to be set in environment."
            )
        return CustomConfig(
            api_key=_attr("CUSTOM_API_KEY", "dummy-key"),
            model=_attr("CUSTOM_MODEL_NAME") or _attr("LLM_MODEL_NAME"),
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    raise ValueError(f"Unhandled provider: {provider}")


# ---------------------------------------------------------------------------
# Helpers (module-private)
# ---------------------------------------------------------------------------

_PLACEHOLDER_PATTERNS = (
    "your_",
    "key_here",
    "<your",
    "sk-xxx",
    "xxxxxxxx",
    "changeme",
)


def _check_placeholder(key: str, provider: str) -> None:
    k = key.strip().lower()
    if not k:
        raise ValueError(f"{provider} api_key must not be empty.")
    if any(k.startswith(p) or p in k for p in _PLACEHOLDER_PATTERNS):
        raise ValueError(
            f"{provider} api_key looks like a placeholder: '{key[:20]}…'. "
            "Set a real API key in the environment."
        )


def mask_api_key(key: str) -> str:
    """Returns a redacted version of an API key safe for logging."""
    if not key:
        return "<EMPTY>"
    k = key.strip()
    if len(k) <= 8:
        return k[:2] + "****"
    return k[:6] + "..." + k[-4:]
