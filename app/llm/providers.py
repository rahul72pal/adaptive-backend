"""
LLM Provider Factory — Production-Level LangChain Integration
=============================================================

Builds a LangChain chat model instance from a validated ``LLMConfig``.
Strict fail-fast: no silent fallbacks. If the configured provider fails,
an exception is raised immediately so the caller knows exactly what went wrong.

Each provider is isolated in its own function to:
  - keep import errors contained (optional deps)
  - make testing easy — mock a single function
  - avoid cross-provider state leakage
"""

from __future__ import annotations

import httpx
from typing import Any

from app.core.logger import logger
from app.llm.config import (
    LLMProvider,
    OpenAIConfig,
    GeminiConfig,
    AnthropicConfig,
    OpenRouterConfig,
    CustomConfig,
    mask_api_key,
)


# ---------------------------------------------------------------------------
# Public factory — single entry-point
# ---------------------------------------------------------------------------

def build_chat_model(
    config: OpenAIConfig | GeminiConfig | AnthropicConfig | OpenRouterConfig | CustomConfig,
    *,
    http_async_client: httpx.AsyncClient | None = None,
    streaming: bool = False,
) -> Any:
    """
    Build and return a LangChain chat model from a validated ``LLMConfig``.

    Parameters
    ----------
    config:
        A validated provider config (discriminated union member).
    http_async_client:
        Optional shared async httpx client for OpenAI-compatible providers.
        When ``None`` a short-lived client is created internally.
    streaming:
        Pass ``True`` when the caller intends to use ``.astream()``/
        ``astream_events()``.

    Raises
    ------
    ImportError
        If the required langchain provider package is not installed.
    ValueError
        If config fields are invalid (validated upstream, but guard here too).
    RuntimeError
        If provider instantiation fails for any other reason.
    """
    provider = LLMProvider(config.provider)
    logger.info(
        "[LLM Provider] Building '%s' | model='%s' | key='%s' | "
        "temperature=%.2f | max_tokens=%d | streaming=%s",
        provider.value,
        config.model,
        mask_api_key(config.api_key),
        config.temperature,
        config.max_tokens,
        streaming,
    )

    if provider == LLMProvider.OPENAI:
        return _build_openai(config, http_async_client=http_async_client, streaming=streaming)  # type: ignore[arg-type]

    if provider == LLMProvider.GEMINI:
        return _build_gemini(config, streaming=streaming)  # type: ignore[arg-type]

    if provider == LLMProvider.ANTHROPIC:
        return _build_anthropic(config, streaming=streaming)  # type: ignore[arg-type]

    if provider == LLMProvider.OPENROUTER:
        return _build_openrouter(config, http_async_client=http_async_client, streaming=streaming)  # type: ignore[arg-type]

    if provider == LLMProvider.CUSTOM:
        return _build_custom(config, http_async_client=http_async_client, streaming=streaming)  # type: ignore[arg-type]

    raise ValueError(f"Unhandled provider: {provider}")


# ---------------------------------------------------------------------------
# Per-provider builders (private)
# ---------------------------------------------------------------------------

def _build_openai(
    config: OpenAIConfig,
    *,
    http_async_client: httpx.AsyncClient | None = None,
    streaming: bool = False,
) -> Any:
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise ImportError(
            "langchain-openai is not installed. Run: pip install langchain-openai"
        ) from exc

    kwargs: dict[str, Any] = dict(
        api_key=config.api_key,
        model=config.model,
        base_url=config.base_url,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        streaming=streaming,
    )
    if http_async_client is not None:
        kwargs["http_async_client"] = http_async_client

    return ChatOpenAI(**kwargs)


def _build_gemini(
    config: GeminiConfig,
    *,
    streaming: bool = False,
) -> Any:
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError as exc:
        raise ImportError(
            "langchain-google-genai is not installed. "
            "Run: pip install langchain-google-genai"
        ) from exc

    return ChatGoogleGenerativeAI(
        google_api_key=config.api_key,
        model=config.model,
        temperature=config.temperature,
        max_output_tokens=config.max_tokens,
        streaming=streaming,
    )


def _build_anthropic(
    config: AnthropicConfig,
    *,
    streaming: bool = False,
) -> Any:
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError as exc:
        raise ImportError(
            "langchain-anthropic is not installed. "
            "Run: pip install langchain-anthropic"
        ) from exc

    return ChatAnthropic(
        api_key=config.api_key,
        model=config.model,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        streaming=streaming,
    )


def _build_openrouter(
    config: OpenRouterConfig,
    *,
    http_async_client: httpx.AsyncClient | None = None,
    streaming: bool = False,
) -> Any:
    """
    OpenRouter API provider (https://openrouter.ai).
    Uses ChatOpenAI configured with OpenRouter's base URL and required headers.
    """
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise ImportError(
            "langchain-openai is not installed. Run: pip install langchain-openai"
        ) from exc

    logger.info("[LLM Provider] OpenRouter model: '%s'", config.model)

    kwargs: dict[str, Any] = dict(
        api_key=config.api_key,
        model=config.model,
        base_url=config.base_url,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        streaming=streaming,
        default_headers={
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "Adaptive Learning Platform",
        },
    )
    if http_async_client is not None:
        kwargs["http_async_client"] = http_async_client

    return ChatOpenAI(**kwargs)


def _build_custom(
    config: CustomConfig,
    *,
    http_async_client: httpx.AsyncClient | None = None,
    streaming: bool = False,
) -> Any:
    """
    Any OpenAI-compatible endpoint (vLLM, Ollama, NVIDIA NIM, LM Studio …).
    Uses ChatOpenAI pointed at the custom base_url.
    """
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise ImportError(
            "langchain-openai is not installed. Run: pip install langchain-openai"
        ) from exc

    logger.info("[LLM Provider] Custom endpoint: '%s'", config.base_url)

    kwargs: dict[str, Any] = dict(
        api_key=config.api_key,
        model=config.model,
        base_url=config.base_url,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        streaming=streaming,
    )
    if http_async_client is not None:
        kwargs["http_async_client"] = http_async_client

    return ChatOpenAI(**kwargs)
