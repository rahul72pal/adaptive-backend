import json
import logging
import re
from typing import Type, Any, Dict, Optional, List
from dataclasses import dataclass, field
import httpx
from pydantic import BaseModel, Field

from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.outputs import LLMResult

from langchain_openai import ChatOpenAI
try:
    from langchain_openrouter import ChatOpenRouter
except ImportError:
    ChatOpenRouter = None

from langchain.agents import create_agent

from app.core.config import Settings, settings
from app.core.logger import logger
from app.utils.exceptions import (
    DeepSeekAPIException,
    DeepSeekTimeoutException,
    RateLimitException,
)
from app.utils.tool import web_search, document_search
from app.services.prompt_service import PromptService


# ------------------------------------------------------------------
# Structured Normalization & Helper Data Structures
# ------------------------------------------------------------------

@dataclass
class NormalizedStreamChunk:
    text: str = ""
    reasoning: str = ""
    tool_call: Optional[Dict[str, Any]] = None


class StreamEventNormalizer:
    """Normalizes provider-specific LangChain stream chunks into unified text, reasoning, and tool_call parts."""

    INTERNAL_MARKERS = [
        "<think>",
        "</think>",
        "<tool_call>",
        "</tool_call>",
        "<function>",
        "</function>",
        "<|DSML|>",
        "<｜DSML｜",
    ]

    @classmethod
    def normalize_chunk(cls, chunk: Any) -> NormalizedStreamChunk:
        norm = NormalizedStreamChunk()
        if not chunk:
            return norm

        content = getattr(chunk, "content", None)

        if isinstance(content, str):
            norm.text = content
        elif isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    if isinstance(block, str):
                        norm.text += block
                    continue

                b_type = block.get("type")
                if b_type == "text":
                    norm.text += block.get("text", "")
                elif b_type in {"reasoning", "thinking", "reasoning_content"}:
                    norm.reasoning += (
                        block.get("text", "")
                        or block.get("reasoning", "")
                        or block.get("content", "")
                    )

        additional_kwargs = getattr(chunk, "additional_kwargs", {}) or {}
        reasoning = (
            additional_kwargs.get("reasoning_content")
            or additional_kwargs.get("reasoning")
            or additional_kwargs.get("reasoning_details")
        )
        if isinstance(reasoning, str):
            norm.reasoning += reasoning

        # Filter out explicit protocol markers and thinking blocks from text
        if norm.text:
            if "<think>" in norm.text or "</think>" in norm.text:
                think_match = re.findall(r'<think>([\s\S]*?)(?:</think>|$)', norm.text, flags=re.IGNORECASE)
                if think_match:
                    norm.reasoning += " ".join(think_match)
                norm.text = clean_thinking_text(norm.text)
            elif "Here's a thinking process:" in norm.text or "Thinking Process:" in norm.text:
                norm.reasoning += norm.text
                norm.text = clean_thinking_text(norm.text)
            else:
                for marker in cls.INTERNAL_MARKERS:
                    if marker in norm.text:
                        norm.text = norm.text.replace(marker, "")

        return norm


def clean_thinking_text(text: str) -> str:
    """Final sanitizer layer to clean leftover reasoning/thinking blocks and protocol markers from complete response."""
    if not text:
        return text

    # 1. Strip complete <think>...</think> tags
    text = re.sub(r'<think>[\s\S]*?</think>', '', text, flags=re.IGNORECASE)
    # 2. Strip unclosed <think>... tags up to end of text
    text = re.sub(r'<think>[\s\S]*$', '', text, flags=re.IGNORECASE)

    # 3. Strip "Here's a thinking process:" or "Thinking Process:" blocks
    text = re.sub(r'(?:Here\'s a thinking process|Thinking Process|\[Thinking Process\])[\s\S]*?(?=\n\n[A-Z0-9]|\n[A-Z0-9]|\Z)', '', text, flags=re.IGNORECASE)

    # 4. Strip protocol markers
    text = re.sub(r'<\|DSML\|>[\s\S]*?</\|DSML\|>', '', text, flags=re.IGNORECASE)
    text = re.sub(r'<function>[\s\S]*?</function>', '', text, flags=re.IGNORECASE)
    text = re.sub(r'<tool_call>[\s\S]*?</tool_call>', '', text, flags=re.IGNORECASE)

    # 5. Strip explicit step-by-step thinking preambles
    if re.search(r'1\.\s*Analyze User Input:', text, flags=re.IGNORECASE):
        match = re.search(r'(?:1\.\s*Analyze User Input:[\s\S]*?)(?:\d+\.\s*(?:Retrieve|Formulate|Structure|Respond)[\s\S]*?\n\n|\n\n(?=[A-Z0-9]))([\s\S]*)', text, flags=re.IGNORECASE)
        if match and match.group(1).strip():
            text = match.group(1).strip()
        else:
            text = re.sub(r'1\.\s*Analyze User Input:[\s\S]*$', '', text, flags=re.IGNORECASE).strip()

    # 6. Strip query rewriter JSON blocks if echoed by agent
    text = re.sub(r'^\s*\{\s*"queries"\s*:\s*\[[\s\S]*?\]\s*,\s*"keywords"\s*:\s*\[[\s\S]*?\]\s*\}\s*', '', text, flags=re.IGNORECASE)

    for marker in ["<think>", "</think>", "<|DSML|>", "</|DSML|>", "<function>", "</function>", "<tool_call>", "</tool_call>"]:
        text = text.replace(marker, "")

    return text.strip()


class TokenAndReasoningCallbackHandler(AsyncCallbackHandler):
    """Callback handler to accumulate token usage and reasoning details across agent steps."""

    def __init__(self) -> None:
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.last_reasoning = None

    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        for generations in response.generations:
            for gen in generations:
                message = getattr(gen, "message", None)
                if message is not None:
                    usage = getattr(message, "usage_metadata", None)
                    if usage is not None:
                        self.total_prompt_tokens += usage.get("input_tokens", 0)
                        self.total_completion_tokens += usage.get("output_tokens", 0)
                        self.total_tokens += usage.get("total_tokens", 0)
                    else:
                        metadata = getattr(message, "response_metadata", {}) or {}
                        if isinstance(metadata, dict):
                            token_usage = metadata.get("token_usage", {}) or metadata.get("usage", {})
                            if isinstance(token_usage, dict):
                                self.total_prompt_tokens += token_usage.get("prompt_tokens", 0) or token_usage.get("input_tokens", 0)
                                self.total_completion_tokens += token_usage.get("completion_tokens", 0) or token_usage.get("output_tokens", 0)
                                self.total_tokens += token_usage.get("total_tokens", 0)

                    reasoning = (
                        message.additional_kwargs.get("reasoning_details")
                        or message.additional_kwargs.get("reasoning_content")
                        or message.additional_kwargs.get("reasoning")
                    )
                    if reasoning is not None:
                        self.last_reasoning = reasoning


# ------------------------------------------------------------------
# LLM Provider Registry Functions
# ------------------------------------------------------------------
PROVIDER_REGISTRY: Dict[str, Any] = {}

def register_provider(name: str):
    def decorator(fn):
        PROVIDER_REGISTRY[name.lower()] = fn
        return fn
    return decorator


@register_provider("anthropic")
def create_anthropic_model(
    *,
    api_key: str,
    model: str,
    temperature: float = 0.7,
    max_tokens: int = 5000,
    http_async_client=None,
    **kwargs
):
    try:
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            api_key=api_key,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except ImportError:
        logger.warning("langchain_anthropic not installed. Falling back to OpenRouter / OpenAI.")
        if ChatOpenRouter is not None:
            return ChatOpenRouter(
                model=model,
                openrouter_api_key=api_key,
                streaming=True,
                temperature=temperature,
                max_tokens=max_tokens,
                request_timeout=600,
            )
        return ChatOpenAI(
            api_key=api_key,
            model=model,
            base_url="https://openrouter.ai/api/v1",
            streaming=True,
            temperature=temperature,
            max_tokens=max_tokens,
            http_async_client=http_async_client,
        )


@register_provider("gemini")
def create_gemini_model(
    *,
    api_key: str,
    model: str,
    temperature: float = 0.7,
    max_tokens: int = 5000,
    http_async_client=None,
    **kwargs
):
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            google_api_key=api_key,
            model=model,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
    except ImportError:
        logger.warning("langchain_google_genai not installed. Falling back to OpenRouter / OpenAI.")
        if ChatOpenRouter is not None:
            return ChatOpenRouter(
                model=model,
                openrouter_api_key=api_key,
                streaming=True,
                temperature=temperature,
                max_tokens=max_tokens,
                request_timeout=600,
            )
        return ChatOpenAI(
            api_key=api_key,
            model=model,
            base_url="https://openrouter.ai/api/v1",
            streaming=True,
            temperature=temperature,
            max_tokens=max_tokens,
            http_async_client=http_async_client,
        )


@register_provider("openai")
def create_openai_model(
    *,
    api_key: str,
    model: str,
    temperature: float = 0.7,
    max_tokens: int = 5000,
    http_async_client=None,
    **kwargs
):
    return ChatOpenAI(
        api_key=api_key,
        model=model,
        streaming=True,
        temperature=temperature,
        max_tokens=max_tokens,
        http_async_client=http_async_client,
    )


@register_provider("openai-compatible")
@register_provider("custom")
def create_openai_compatible_model(
    *,
    api_key: str,
    model: str,
    base_url: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 5000,
    http_async_client=None,
    **kwargs
):
    if base_url:
        return ChatOpenAI(
            api_key=api_key,
            model=model,
            base_url=base_url,
            streaming=True,
            temperature=temperature,
            max_tokens=max_tokens,
            http_async_client=http_async_client,
        )
    return ChatOpenAI(
        api_key=api_key,
        model=model,
        streaming=True,
        temperature=temperature,
        max_tokens=max_tokens,
        http_async_client=http_async_client,
    )


@register_provider("openrouter")
def create_openrouter_model(
    *,
    api_key: str,
    model: str,
    temperature: float = 0.7,
    max_tokens: int = 5000,
    http_async_client=None,
    **kwargs
):
    if ChatOpenRouter is not None:
        return ChatOpenRouter(
            model=model,
            openrouter_api_key=api_key,
            streaming=True,
            temperature=temperature,
            max_tokens=max_tokens,
            request_timeout=600,
        )
    return ChatOpenAI(
        api_key=api_key,
        model=model,
        base_url="https://openrouter.ai/api/v1",
        streaming=True,
        temperature=temperature,
        max_tokens=max_tokens,
        http_async_client=http_async_client,
    )


# ------------------------------------------------------------------
# Agent Service Execution Flow
# ------------------------------------------------------------------

class AgentService:
    """Service to handle agent creation and communication using dynamic User LLM Configuration & System Prompts."""

    def __init__(self, client: Optional[httpx.AsyncClient] = None, settings_obj: Optional[Settings] = None) -> None:
        self.client = client or httpx.AsyncClient(timeout=httpx.Timeout(connect=15.0, read=None, write=30.0, pool=30.0))
        self.settings = settings_obj or settings
        self.prompt_service = PromptService()
        self.default_system_prompt = self.prompt_service.get_system_prompt()
        self.tools = [web_search, document_search]

    @property
    def system_prompt(self) -> str:
        """Exposes the default system prompt for fallback compatibility."""
        return self.default_system_prompt

    def _get_llm_model(self, llm_config: Optional[Dict[str, Any]] = None):
        """Constructs an LLM model instance using registered provider functions."""
        llm_async_client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=15.0, read=None, write=30.0, pool=30.0)
        )
        logger.info(f"LLM Config==> {llm_config}")

        if not llm_config or not llm_config.get("api_key"):
            provider = (getattr(self.settings, "LLM_PROVIDER", "") or "openai").lower()
            if provider == "gemini" and getattr(self.settings, "GEMINI_API_KEY", ""):
                llm_config = {
                    "config_type": "provider",
                    "provider": "gemini",
                    "api_key": self.settings.GEMINI_API_KEY,
                    "default_model": self.settings.GEMINI_MODEL_NAME,
                    "max_tokens": getattr(self.settings, "LLM_MAX_TOKENS", 5000),
                }
            elif provider == "openrouter" and getattr(self.settings, "OPENROUTER_API_KEY", ""):
                llm_config = {
                    "config_type": "provider",
                    "provider": "openrouter",
                    "api_key": self.settings.OPENROUTER_API_KEY,
                    "default_model": getattr(self.settings, "OPENROUTER_MODEL_NAME", self.settings.LLM_MODEL_NAME),
                    "max_tokens": getattr(self.settings, "LLM_MAX_TOKENS", 5000),
                }
            elif provider == "custom":
                import os
                custom_url = getattr(self.settings, "CUSTOM_BASE_URL", "") or getattr(self.settings, "CUSTOM_API_BASE", "") or os.getenv("CUSTOM_BASE_URL", "") or os.getenv("CUSTOM_API_BASE", "")
                custom_key = getattr(self.settings, "CUSTOM_API_KEY", "") or os.getenv("CUSTOM_API_KEY", "") or "dummy-key"
                llm_config = {
                    "config_type": "custom",
                    "base_url": custom_url,
                    "api_key": custom_key,
                    "model": getattr(self.settings, "CUSTOM_MODEL_NAME", "") or self.settings.LLM_MODEL_NAME,
                    "max_tokens": getattr(self.settings, "LLM_MAX_TOKENS", 5000),
                }
            elif getattr(self.settings, "OPENAI_API_KEY", ""):
                llm_config = {
                    "config_type": "provider",
                    "provider": "openai",
                    "api_key": self.settings.OPENAI_API_KEY,
                    "default_model": self.settings.LLM_MODEL_NAME,
                    "max_tokens": getattr(self.settings, "LLM_MAX_TOKENS", 5000),
                }

        if not llm_config or not llm_config.get("api_key"):
            raise DeepSeekAPIException(
                message="LLM configuration not found for user. Please configure your LLM API Key and Model at /llm-config.",
                status_code=400
            )

        config_type = llm_config.get("config_type", "provider")
        api_key = llm_config.get("api_key", "").strip()
        if not api_key:
            raise DeepSeekAPIException(
                message="API key missing in user LLM configuration.",
                status_code=400
            )

        temperature = float(llm_config.get("temperature", 0.7)) if llm_config.get("temperature") is not None else 0.7
        max_tokens = int(llm_config.get("max_tokens", 5000)) if llm_config.get("max_tokens") is not None else 5000

        if config_type == "provider":
            provider_name = (llm_config.get("provider") or "openrouter").lower()
            model_name = llm_config.get("default_model") or llm_config.get("model")
            if not model_name:
                raise DeepSeekAPIException(
                    message="Model name missing in user LLM configuration.",
                    status_code=400
                )

            creator_fn = PROVIDER_REGISTRY.get(provider_name, PROVIDER_REGISTRY.get("openai"))
            logger.info(f"Instantiating model using registered provider '{provider_name}' | model: '{model_name}' | temp: {temperature} | max_tokens: {max_tokens}")
            return creator_fn(
                api_key=api_key,
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                http_async_client=llm_async_client
            )

        elif config_type == "custom":
            base_url = llm_config.get("base_url")
            model_name = llm_config.get("model")
            if not base_url or not model_name:
                raise DeepSeekAPIException(
                    message="Custom LLM configuration requires both 'base_url' and 'model'.",
                    status_code=400
                )

            creator_fn = PROVIDER_REGISTRY.get("openai-compatible")
            logger.info(f"Instantiating openai-compatible model for custom endpoint: '{base_url}' | model: '{model_name}' | temp: {temperature} | max_tokens: {max_tokens}")
            return creator_fn(
                api_key=api_key,
                model=model_name,
                base_url=base_url,
                temperature=temperature,
                max_tokens=max_tokens,
                http_async_client=llm_async_client
            )
        else:
            raise DeepSeekAPIException(
                message=f"Unsupported configuration type '{config_type}'.",
                status_code=400
            )

    def _format_system_prompt(self, prompt: str) -> str:
        """Dynamically injects current date and time reference into system prompt."""
        from datetime import datetime
        try:
            import pytz
            now = datetime.now(pytz.timezone("Asia/Kolkata"))
        except Exception:
            now = datetime.utcnow()

        current_time_str = now.strftime("%A, %d %B %Y, %I:%M %p %Z")

        if "{current_time_str}" in prompt:
            return prompt.replace("{current_time_str}", current_time_str)

        if "Current Date & Time Reference:" not in prompt and "Current Time:" not in prompt:
            return f"{prompt}\n\n[Current Date & Time Reference: {current_time_str}]"

        return prompt

    def _create_agent_executor(
        self,
        llm_config: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None,
        behavior: Optional[str] = None
    ):
        """Compiles an agent executor with specified model configuration and system prompt."""
        custom_behavior = system_prompt if system_prompt is not None else behavior
        model = self._get_llm_model(llm_config)
        prompt = self.prompt_service.get_system_prompt(behavior=custom_behavior)
        formatted_prompt = self._format_system_prompt(prompt)

        try:
            from langchain.agents.middleware import SummarizationMiddleware
            middleware_list = [
                SummarizationMiddleware(
                    model=model,
                    trigger=("messages", 50),
                    keep=("messages", 20)
                )
            ]
        except Exception as mw_err:
            logger.warning(f"SummarizationMiddleware import warning: {mw_err}")
            middleware_list = []

        return create_agent(
            model=model,
            tools=self.tools,
            system_prompt=formatted_prompt,
            middleware=middleware_list,
            debug=False
        )

    def _convert_messages(self, messages: List[Dict[str, Any]]) -> List[Any]:
        lc_messages = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")
            reasoning_details = msg.get("reasoning_details")

            if role == "user":
                lc_messages.append(HumanMessage(content=content))
            elif role == "system":
                lc_messages.append(SystemMessage(content=content))
            elif role == "assistant":
                kwargs = {}
                if reasoning_details is not None:
                    kwargs["reasoning_details"] = reasoning_details
                lc_messages.append(AIMessage(content=content, additional_kwargs=kwargs))
        return lc_messages

    async def astream_chat_response(
        self,
        messages: List[Dict[str, Any]],
        llm_config: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None
    ):
        """Streams multi-event updates (tool execution status, reasoning deltas, text deltas, final result)."""
        agent_executor = self._create_agent_executor(llm_config=llm_config, system_prompt=system_prompt)
        lc_messages = self._convert_messages(messages)
        callback_handler = TokenAndReasoningCallbackHandler()

        full_content = ""
        full_reasoning = ""
        active_tool_depth = 0

        # Reset document search attempt counter at start of turn
        from app.utils.tool import reset_search_counter
        reset_search_counter()

        try:
            logger.info("Streaming agent execution events...")
            async for event in agent_executor.astream_events(
                {"messages": lc_messages},
                version="v2",
                config={"callbacks": [callback_handler], "recursion_limit": 10}
            ):
                event_kind = event.get("event")
                event_name = event.get("name", "")

                if event_kind == "on_tool_start":
                    active_tool_depth += 1
                    tool_input = event.get("data", {}).get("input")
                    query_str = ""
                    if isinstance(tool_input, dict):
                        query_str = tool_input.get("query") or tool_input.get("input") or str(tool_input)
                    elif isinstance(tool_input, str):
                        query_str = tool_input

                    user_status = f"Searching using {event_name}..."
                    if event_name == "web_search":
                        user_status = f"Searching web for '{query_str}'..." if query_str else "Searching web..."
                    elif event_name == "document_search":
                        user_status = f"Searching knowledge base for '{query_str}'..." if query_str else "Searching knowledge base..."

                    yield {
                        "type": "status",
                        "status": "searching",
                        "tool": event_name,
                        "message": user_status,
                        "query": query_str
                    }

                elif event_kind == "on_tool_end":
                    active_tool_depth = max(0, active_tool_depth - 1)
                    yield {
                        "type": "status",
                        "status": "tool_completed",
                        "tool": event_name,
                        "message": f"Finished {event_name} execution"
                    }

                elif event_kind == "on_chat_model_stream":
                    # Ignore LLM stream deltas emitted by inner tool models while a tool is executing
                    if active_tool_depth > 0:
                        continue

                    chunk = event.get("data", {}).get("chunk")
                    if chunk:
                        norm = StreamEventNormalizer.normalize_chunk(chunk)

                        if norm.text:
                            full_content += norm.text
                            yield {"type": "text_delta", "delta": norm.text}

                        if norm.reasoning:
                            full_reasoning += norm.reasoning

                elif event_kind == "on_chat_model_end":
                    output = event.get("data", {}).get("output")
                    if output:
                        usage = getattr(output, "usage_metadata", None)
                        if usage and isinstance(usage, dict):
                            callback_handler.total_prompt_tokens += usage.get("input_tokens", 0)
                            callback_handler.total_completion_tokens += usage.get("output_tokens", 0)
                            callback_handler.total_tokens += usage.get("total_tokens", 0)
                        else:
                            meta = getattr(output, "response_metadata", {}) or {}
                            toks = meta.get("token_usage", {}) or meta.get("usage", {})
                            if toks and isinstance(toks, dict):
                                callback_handler.total_prompt_tokens += toks.get("prompt_tokens", 0) or toks.get("input_tokens", 0)
                                callback_handler.total_completion_tokens += toks.get("completion_tokens", 0) or toks.get("output_tokens", 0)
                                callback_handler.total_tokens += toks.get("total_tokens", 0)

            # Fallback calculation if provider does not report token usage metadata in streaming
            if callback_handler.total_prompt_tokens == 0:
                estimated_prompt = sum(len(str(m.get("content", ""))) // 4 for m in messages)
                callback_handler.total_prompt_tokens = max(12, estimated_prompt)

            if callback_handler.total_completion_tokens == 0:
                estimated_completion = len(full_content) // 4
                callback_handler.total_completion_tokens = max(5, estimated_completion)

            callback_handler.total_tokens = callback_handler.total_prompt_tokens + callback_handler.total_completion_tokens

            usage_dict = {
                "prompt_tokens": callback_handler.total_prompt_tokens,
                "completion_tokens": callback_handler.total_completion_tokens,
                "total_tokens": callback_handler.total_tokens
            }

            clean_content = clean_thinking_text(full_content)

            yield {
                "type": "done",
                "message": clean_content,
                "reasoning_details": callback_handler.last_reasoning or full_reasoning or None,
                "usage": usage_dict
            }

        except Exception as exc:
            logger.exception(f"Error in astream_chat_response: {exc}")
            err_str = str(exc)
            if "Recursion limit" in err_str or "GraphRecursionError" in str(type(exc)):
                fallback_msg = "Information search reached maximum retry attempts without finding conclusive facts. Please refine your query or verify uploaded documents."
            else:
                fallback_msg = "Server issue: Please contact the chatbot owner."
            yield {
                "type": "text_delta",
                "delta": fallback_msg
            }
            yield {
                "type": "done",
                "message": fallback_msg,
                "reasoning_details": None,
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            }

    async def get_chat_response(
        self,
        messages: List[Dict[str, Any]],
        llm_config: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Calls the agent workflow and returns final response message and metadata."""
        try:
            agent_executor = self._create_agent_executor(llm_config=llm_config, system_prompt=system_prompt)
            lc_messages = self._convert_messages(messages)

            logger.info("Invoking compiled agent workflow...")
            callback_handler = TokenAndReasoningCallbackHandler()

            response = await agent_executor.ainvoke(
                {"messages": lc_messages},
                config={"callbacks": [callback_handler], "recursion_limit": 10}
            )

            if isinstance(response, dict) and "messages" in response:
                final_message = response["messages"][-1]
                content = final_message.content
            else:
                content = getattr(response, "output", str(response))

            content = clean_thinking_text(content)

            if callback_handler.total_prompt_tokens == 0:
                estimated_prompt = sum(len(str(m.get("content", ""))) // 4 for m in messages)
                callback_handler.total_prompt_tokens = max(12, estimated_prompt)

            if callback_handler.total_completion_tokens == 0:
                estimated_completion = len(content) // 4
                callback_handler.total_completion_tokens = max(5, estimated_completion)

            callback_handler.total_tokens = callback_handler.total_prompt_tokens + callback_handler.total_completion_tokens

            usage_dict = {
                "prompt_tokens": callback_handler.total_prompt_tokens,
                "completion_tokens": callback_handler.total_completion_tokens,
                "total_tokens": callback_handler.total_tokens
            }

            return {
                "message": content,
                "usage": usage_dict,
                "reasoning_details": callback_handler.last_reasoning
            }

        except Exception as exc:
            logger.exception(f"Unexpected error in get_chat_response: {exc}")
            fallback_msg = "Server issue: Please contact the chatbot owner."
            return {
                "message": fallback_msg,
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "reasoning_details": None
            }


# ------------------------------------------------------------------
# Structured JSON LLM Gateway (For Engine Graphs & Direct Schemas)
# ------------------------------------------------------------------

def parse_json_from_llm_response(text: str, schema: Type[BaseModel], topic_hint: str = "") -> BaseModel:
    """Extracts valid JSON payload safely, ignoring conversational preambles and thinking blocks."""
    if not text:
        raise ValueError("Empty LLM response received")

    logger.info("[LLM Gateway] Raw response type: %s", type(text))
    logger.info("[LLM Gateway] Raw response repr: %r", text[:5000])
    logger.info("[LLM Gateway] Response starts with: %r", text[:100])
    logger.info("[LLM Gateway] Response ends with: %r", text[-100:])

    # 1. Clean thinking blocks and protocol markers
    content = clean_thinking_text(text).strip()

    # 2. Remove markdown code fences if present
    if content.startswith("```json"):
        content = content[7:].strip()
    elif content.startswith("```"):
        content = content[3:].strip()

    if content.endswith("```"):
        content = content[:-3].strip()

    if "```json" in content:
        content = content.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in content:
        content = content.split("```", 1)[1].split("```", 1)[0].strip()

    # 3. Direct json.loads attempt
    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict):
            return schema.model_validate(parsed)
        elif isinstance(parsed, list):
            dict_data = {"topic": topic_hint or "DSA", "nodes": parsed, "concepts": parsed}
            return schema.model_validate(dict_data)
    except Exception as e:
        logger.debug(f"[LLM Gateway] Direct json.loads failed ({e}), trying candidate extraction...")

    # 4. Substring candidate extraction: first '{' to last '}'
    start = content.find('{')
    end = content.rfind('}')

    if start != -1 and end != -1 and end > start:
        candidate = content[start:end + 1]
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return schema.model_validate(parsed)
        except Exception as e:
            logger.debug(f"[LLM Gateway] First-to-last brace json.loads failed ({e})...")

    # 5. Candidate extraction: raw_decode starting at first '{'
    if start != -1:
        substring = content[start:]
        try:
            parsed, _ = json.JSONDecoder().raw_decode(substring)
            if isinstance(parsed, dict):
                return schema.model_validate(parsed)
        except Exception:
            pass

    # 6. Candidate extraction: check raw array [...]
    first_arr = content.find('[')
    last_arr = content.rfind(']')
    if first_arr != -1 and last_arr > first_arr:
        arr_candidate = content[first_arr:last_arr + 1]
        try:
            parsed = json.loads(arr_candidate)
            if isinstance(parsed, list):
                dict_data = {"topic": topic_hint or "DSA", "nodes": parsed, "concepts": parsed}
                return schema.model_validate(dict_data)
        except Exception:
            pass

    # 7. Candidate extraction: Search all '{' occurrences
    for match in re.finditer(r'\{', content):
        s_pos = match.start()
        try:
            parsed, _ = json.JSONDecoder().raw_decode(content[s_pos:])
            if isinstance(parsed, dict):
                return schema.model_validate(parsed)
        except Exception:
            continue

    raise ValueError(
        f"No JSON object could be extracted from response: {text[:2000]}"
    )




def _is_placeholder_key(key: str) -> bool:
    if not key:
        return True
    k = key.strip().lower()
    return k.startswith("your_") or "key_here" in k or k == "dummy-key"


def _mask_api_key(key: str) -> str:
    if not key:
        return "<EMPTY>"
    k = key.strip()
    if len(k) <= 8:
        return k[:2] + "****" + k[-2:]
    return k[:6] + "..." + k[-4:]


async def call_llm_gateway(
    prompt: ChatPromptTemplate,
    params: Dict[str, Any],
    schema: Type[BaseModel]
) -> BaseModel:
    """Invokes LLM Gateway using strictly configured provider and model name without hardcoded fallbacks."""
    parser = PydanticOutputParser(pydantic_object=schema)
    fmt_inst = parser.get_format_instructions()
    strict_fmt = f"{fmt_inst}\nCRITICAL REQUIREMENT: Return ONLY a raw valid JSON object complying with the schema above. Do NOT output any introductory text, preambles, thinking, or explanations outside the JSON object."
    formatted_prompt = prompt.partial(format_instructions=strict_fmt)
    last_error: Optional[Exception] = None
    max_tokens = getattr(settings, "LLM_MAX_TOKENS", 5000)
    preferred_provider = (getattr(settings, "LLM_PROVIDER", "") or "").strip().lower()

    logger.info(f"[LLM Gateway] Triggered for schema='{schema.__name__}' | Target Provider='{preferred_provider}'")

    # --- Provider Helpers ---

    async def try_gemini() -> Optional[BaseModel]:
        nonlocal last_error
        gemini_key = (getattr(settings, "GEMINI_API_KEY", "") or "").strip()
        if _is_placeholder_key(gemini_key):
            logger.warning("[LLM Gateway] Gemini skipped: Placeholder or missing API key.")
            return None
        model_name = (getattr(settings, "GEMINI_MODEL_NAME", "") or getattr(settings, "LLM_MODEL_NAME", "")).strip()
        if not model_name:
            logger.warning("[LLM Gateway] Gemini skipped: No model name specified.")
            return None

        logger.info(f"[LLM Gateway] Initializing Gemini | Provider: 'gemini' | Model: '{model_name}' | API Key: '{_mask_api_key(gemini_key)}'")

        from langchain_google_genai import ChatGoogleGenerativeAI
        try:
            gemini_model = ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=gemini_key,
                temperature=0.2,
                max_output_tokens=max_tokens
            )
            try:
                chain = formatted_prompt | gemini_model | parser
                return await chain.ainvoke(params)
            except Exception as invoke_err:
                err_str = str(invoke_err)
                if any(code in err_str for code in ["401", "403", "404", "API_KEY_INVALID", "RESOURCE_EXHAUSTED"]):
                    raise invoke_err
                raw_chain = formatted_prompt | gemini_model
                res = await raw_chain.ainvoke(params)
                content = res.content if hasattr(res, "content") else str(res)
                return parse_json_from_llm_response(content, schema)
        except Exception as e:
            last_error = e
            logger.warning(f"[LLM Gateway] Gemini model '{model_name}' failed ({e}).")
        return None

    async def try_openai() -> Optional[BaseModel]:
        nonlocal last_error
        openai_key = (getattr(settings, "OPENAI_API_KEY", "") or "").strip()
        if _is_placeholder_key(openai_key):
            logger.warning("[LLM Gateway] OpenAI skipped: Placeholder or missing API key.")
            return None
        model_name = (getattr(settings, "OPENAI_MODEL_NAME", "") or getattr(settings, "LLM_MODEL_NAME", "")).strip()
        if not model_name:
            logger.warning("[LLM Gateway] OpenAI skipped: No model name specified.")
            return None

        base_url = (getattr(settings, "OPENAI_API_BASE", None) or "https://api.openai.com/v1")
        logger.info(f"[LLM Gateway] Initializing OpenAI | Provider: 'openai' | Model: '{model_name}' | API Key: '{_mask_api_key(openai_key)}' | Base URL: '{base_url}'")

        from langchain_openai import ChatOpenAI
        try:
            llm = ChatOpenAI(
                model=model_name,
                openai_api_key=openai_key,
                openai_api_base=base_url,
                temperature=0.2,
                max_tokens=max_tokens
            )
            try:
                chain = formatted_prompt | llm | parser
                return await chain.ainvoke(params)
            except Exception as invoke_err:
                err_str = str(invoke_err)
                if any(code in err_str for code in ["401", "402", "403", "404"]):
                    raise invoke_err
                raw_chain = formatted_prompt | llm
                res = await raw_chain.ainvoke(params)
                content = res.content if hasattr(res, "content") else str(res)
                return parse_json_from_llm_response(content, schema)
        except Exception as e:
            last_error = e
            logger.warning(f"[LLM Gateway] OpenAI model '{model_name}' failed ({e}).")
        return None

    async def try_openrouter() -> Optional[BaseModel]:
        nonlocal last_error
        openrouter_key = (getattr(settings, "OPENROUTER_API_KEY", "") or "").strip()
        if _is_placeholder_key(openrouter_key):
            logger.warning("[LLM Gateway] OpenRouter skipped: Placeholder or missing API key.")
            return None
        model_name = (getattr(settings, "OPENROUTER_MODEL_NAME", "") or getattr(settings, "LLM_MODEL_NAME", "")).strip()
        if not model_name:
            logger.warning("[LLM Gateway] OpenRouter skipped: No model name specified.")
            return None

        logger.info(f"[LLM Gateway] Initializing OpenRouter | Provider: 'openrouter' | Model: '{model_name}' | API Key: '{_mask_api_key(openrouter_key)}'")

        headers = {
            "HTTP-Referer": getattr(settings, "APP_URL", "http://localhost:8000"),
            "X-Title": "Adaptive Learning Platform"
        }

        try:
            try:
                from langchain_openrouter import ChatOpenRouter
                llm = ChatOpenRouter(
                    model=model_name,
                    openrouter_api_key=openrouter_key,
                    temperature=0.2,
                    max_tokens=max_tokens,
                )
            except ImportError:
                from langchain_openai import ChatOpenAI
                llm = ChatOpenAI(
                    model=model_name,
                    openai_api_key=openrouter_key,
                    openai_api_base="https://openrouter.ai/api/v1",
                    default_headers=headers,
                    temperature=0.2,
                    max_tokens=max_tokens
                )

            try:
                chain = formatted_prompt | llm | parser
                return await chain.ainvoke(params)
            except Exception as invoke_err:
                err_str = str(invoke_err)
                if any(code in err_str for code in ["401", "402", "403", "404"]):
                    raise invoke_err
                raw_chain = formatted_prompt | llm
                res = await raw_chain.ainvoke(params)
                content = res.content if hasattr(res, "content") else str(res)
                return parse_json_from_llm_response(content, schema)
        except Exception as e:
            last_error = e
            logger.warning(f"[LLM Gateway] OpenRouter model '{model_name}' failed ({e}).")
        return None

    async def try_groq() -> Optional[BaseModel]:
        nonlocal last_error
        groq_key = (getattr(settings, "GROQ_API_KEY", "") or "").strip()
        if _is_placeholder_key(groq_key):
            logger.warning("[LLM Gateway] Groq skipped: Placeholder or missing API key.")
            return None
        model_name = (getattr(settings, "GROQ_MODEL_NAME", "") or getattr(settings, "LLM_MODEL_NAME", "")).strip()
        if not model_name:
            logger.warning("[LLM Gateway] Groq skipped: No model name specified.")
            return None

        logger.info(f"[LLM Gateway] Initializing Groq | Provider: 'groq' | Model: '{model_name}' | API Key: '{_mask_api_key(groq_key)}'")

        from langchain_groq import ChatGroq
        try:
            groq_model = ChatGroq(
                model_name=model_name,
                groq_api_key=groq_key,
                temperature=0.2,
                max_tokens=max_tokens
            )
            try:
                fallback_chain = formatted_prompt | groq_model | parser
                return await fallback_chain.ainvoke(params)
            except Exception as invoke_err:
                err_str = str(invoke_err)
                if any(code in err_str for code in ["401", "403", "404", "model_decommissioned", "model_not_found"]):
                    raise invoke_err
                raw_chain = formatted_prompt | groq_model
                res = await raw_chain.ainvoke(params)
                content = res.content if hasattr(res, "content") else str(res)
                return parse_json_from_llm_response(content, schema)
        except Exception as e:
            last_error = e
            logger.warning(f"[LLM Gateway] Groq model '{model_name}' failed ({e}).")
        return None

    async def try_custom() -> Optional[BaseModel]:
        """Custom provider for OpenAI-compatible gateways (e.g. vLLM, Ollama, LocalAI, NVIDIA NIM)."""
        nonlocal last_error
        import os

        url_candidate = getattr(settings, "CUSTOM_BASE_URL", "") or getattr(settings, "CUSTOM_API_BASE", "") or os.getenv("CUSTOM_BASE_URL", "") or os.getenv("CUSTOM_API_BASE", "")
        key_candidate = getattr(settings, "CUSTOM_API_KEY", "") or os.getenv("CUSTOM_API_KEY", "")

        # If url_candidate is not a valid URL (e.g. key was passed in CUSTOM_API_BASE), fix it
        if url_candidate and not url_candidate.startswith("http"):
            if not key_candidate or _is_placeholder_key(key_candidate):
                key_candidate = url_candidate
            url_candidate = getattr(settings, "CUSTOM_BASE_URL", "") if getattr(settings, "CUSTOM_BASE_URL", "").startswith("http") else ""

        base_url = url_candidate.strip()
        if not base_url or not base_url.startswith("http"):
            logger.warning("[LLM Gateway] Custom skipped: Invalid or missing Base URL (must start with http:// or https://).")
            return None

        api_key = (key_candidate or "dummy-key").strip()
        model_name = (getattr(settings, "CUSTOM_MODEL_NAME", "") or getattr(settings, "LLM_MODEL_NAME", "")).strip()
        if not model_name:
            logger.warning("[LLM Gateway] Custom skipped: No model name specified.")
            return None

        logger.info(f"[LLM Gateway] Initializing Custom | Provider: 'custom' | Model: '{model_name}' | API Key: '{_mask_api_key(api_key)}' | Base URL: '{base_url}'")

        from langchain_openai import ChatOpenAI
        topic_hint = str(params.get("topic", ""))

        # 1. Attempt json_object mode via model_kwargs if supported by endpoint/NIM
        try:
            llm_json = ChatOpenAI(
                model=model_name,
                openai_api_key=api_key,
                openai_api_base=base_url,
                temperature=0.2,
                max_tokens=max_tokens,
                request_timeout=25.0,
                model_kwargs={"response_format": {"type": "json_object"}}
            )
            raw_chain = formatted_prompt | llm_json
            res = await raw_chain.ainvoke(params)
            content = res.content if hasattr(res, "content") else str(res)
            return parse_json_from_llm_response(content, schema, topic_hint=topic_hint)
        except Exception as json_mode_err:
            logger.debug(f"[LLM Gateway] Custom json_object mode fallback ({json_mode_err}), trying standard ChatOpenAI...")

        # 2. Standard ChatOpenAI call with Pydantic parser & multi-stage JSON extractor
        try:
            llm = ChatOpenAI(
                model=model_name,
                openai_api_key=api_key,
                openai_api_base=base_url,
                temperature=0.2,
                max_tokens=max_tokens,
                request_timeout=25.0
            )
            try:
                chain = formatted_prompt | llm | parser
                return await chain.ainvoke(params)
            except Exception as invoke_err:
                err_str = str(invoke_err)
                if any(code in err_str for code in ["401", "402", "403", "404"]):
                    raise invoke_err
                raw_chain = formatted_prompt | llm
                res = await raw_chain.ainvoke(params)
                content = res.content if hasattr(res, "content") else str(res)
                return parse_json_from_llm_response(content, schema, topic_hint=topic_hint)
        except Exception as e:
            last_error = e
            logger.warning(f"[LLM Gateway] Custom LLM model '{model_name}' at endpoint '{base_url}' failed: {e}")
        return None


    # --- Match / Switch Case Selection ---

    match preferred_provider:
        case "openai":
            primary_func = try_openai
        case "openrouter":
            primary_func = try_openrouter
        case "gemini":
            primary_func = try_gemini
        case "groq":
            primary_func = try_groq
        case "custom":
            primary_func = try_custom
        case _:
            primary_func = None

    all_funcs = [try_gemini, try_groq, try_openai, try_openrouter, try_custom]
    if primary_func:
        execution_order = [primary_func] + [f for f in all_funcs if f != primary_func]
    else:
        execution_order = all_funcs

    for func in execution_order:
        res = await func()
        if res is not None:
            return res

    provider_desc = preferred_provider if preferred_provider else "configured"
    err_detail = f"Details: {last_error}" if last_error else f"No valid API key or endpoint configured for provider '{provider_desc}'."
    raise RuntimeError(f"LLM Provider '{provider_desc}' failed for schema '{schema.__name__}'. {err_detail}")