"""
LLM Gateway — Production-Level LangChain / LangGraph Integration
================================================================

Architecture
------------
  LLMConfig (validated)
       │
       ▼
  build_chat_model()   ←── app/llm/providers.py
       │
       ▼
  AgentService          ←── streaming & non-streaming agent execution
       │
       ├── astream_chat_response()    →  async generator of typed events
       └── get_chat_response()        →  single response dict

  call_llm_gateway()               →  structured JSON call (Pydantic schema)
  call_llm_gateway_stream()        →  streaming text delta generator

Design decisions
----------------
* NO silent fallbacks — if the configured provider fails, the error is raised
  immediately so callers receive a deterministic 5xx instead of a mystery
  "fallback" response from an unexpected model.
* Structured output uses ``parse_json_from_llm_response`` with ``attempt_repair_json``
  as a single recovery layer, then fails hard.
* All provider configuration comes from ``LLMConfig`` Pydantic objects —
  no raw env-var reads inside this file.
* LangGraph ``create_react_agent`` is used for the agent executor (LangGraph-native,
  replaces the deprecated ``langchain.agents.create_agent``).
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type, Union

import httpx
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.callbacks import AsyncCallbackHandler
from langchain_core.outputs import LLMResult
from pydantic import BaseModel

from app.core.config import Settings, settings
from app.core.logger import logger
from app.llm.config import (
    AnthropicConfig,
    CustomConfig,
    GeminiConfig,
    OpenAIConfig,
    OpenRouterConfig,
    llm_config_from_dict,
    llm_config_from_settings,
    mask_api_key,
)
from app.llm.providers import build_chat_model
from app.llm.user_context import get_active_llm_config
from app.utils.exceptions import DeepSeekAPIException
from app.utils.tool import document_search, reset_search_counter, web_search
from app.services.prompt_service import PromptService


# ===========================================================================
# Stream normalisation helpers
# ===========================================================================

@dataclass
class NormalizedStreamChunk:
    text: str = ""
    reasoning: str = ""
    tool_call: Optional[Dict[str, Any]] = None


_INTERNAL_MARKERS = (
    "<think>", "</think>",
    "<tool_call>", "</tool_call>",
    "<function>", "</function>",
    "<|DSML|>", "<｜DSML｜",
)


class StreamEventNormalizer:
    """Normalises provider-specific LangChain stream chunks into text/reasoning/tool_call."""

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

        if norm.text:
            if "<think>" in norm.text or "</think>" in norm.text:
                think_match = re.findall(
                    r"<think>([\s\S]*?)(?:</think>|$)", norm.text, flags=re.IGNORECASE
                )
                if think_match:
                    norm.reasoning += " ".join(think_match)
                norm.text = clean_thinking_text(norm.text)
            else:
                for marker in _INTERNAL_MARKERS:
                    if marker in norm.text:
                        norm.text = norm.text.replace(marker, "")

        return norm


def clean_thinking_text(text: str) -> str:
    """Remove reasoning/thinking blocks and protocol markers from LLM response text."""
    if not text:
        return text

    text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<think>[\s\S]*$", "", text, flags=re.IGNORECASE)
    text = re.sub(
        r"(?:Here's a thinking process|Thinking Process|\[Thinking Process\])[\s\S]*?"
        r"(?=\n\n[A-Z0-9]|\n[A-Z0-9]|\Z)",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"<\|DSML\|>[\s\S]*?</\|DSML\|>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<function>[\s\S]*?</function>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<tool_call>[\s\S]*?</tool_call>", "", text, flags=re.IGNORECASE)

    if re.search(r"1\.\s*Analyze User Input:", text, flags=re.IGNORECASE):
        match = re.search(
            r"(?:1\.\s*Analyze User Input:[\s\S]*?)(?:\d+\.\s*(?:Retrieve|Formulate|Structure|Respond)"
            r"[\s\S]*?\n\n|\n\n(?=[A-Z0-9]))([\s\S]*)",
            text,
            flags=re.IGNORECASE,
        )
        if match and match.group(1).strip():
            text = match.group(1).strip()
        else:
            text = re.sub(r"1\.\s*Analyze User Input:[\s\S]*$", "", text, flags=re.IGNORECASE).strip()

    text = re.sub(
        r'^\s*\{\s*"queries"\s*:\s*\[[\s\S]*?\]\s*,\s*"keywords"\s*:\s*\[[\s\S]*?\]\s*\}\s*',
        "",
        text,
        flags=re.IGNORECASE,
    )

    for marker in ("<think>", "</think>", "<|DSML|>", "</|DSML|>", "<function>", "</function>", "<tool_call>", "</tool_call>"):
        text = text.replace(marker, "")

    return text.strip()


# ===========================================================================
# Token & reasoning tracking callback
# ===========================================================================

class TokenAndReasoningCallbackHandler(AsyncCallbackHandler):
    """Accumulates token usage and reasoning content across agent steps."""

    def __init__(self) -> None:
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.last_reasoning: Optional[Any] = None

    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        for generations in response.generations:
            for gen in generations:
                message = getattr(gen, "message", None)
                if message is None:
                    continue
                usage = getattr(message, "usage_metadata", None)
                if usage is not None:
                    self.total_prompt_tokens += usage.get("input_tokens", 0)
                    self.total_completion_tokens += usage.get("output_tokens", 0)
                    self.total_tokens += usage.get("total_tokens", 0)
                else:
                    meta = getattr(message, "response_metadata", {}) or {}
                    if isinstance(meta, dict):
                        toks = meta.get("token_usage", {}) or meta.get("usage", {})
                        if isinstance(toks, dict):
                            self.total_prompt_tokens += toks.get("prompt_tokens", 0) or toks.get("input_tokens", 0)
                            self.total_completion_tokens += toks.get("completion_tokens", 0) or toks.get("output_tokens", 0)
                            self.total_tokens += toks.get("total_tokens", 0)

                reasoning = (
                    message.additional_kwargs.get("reasoning_details")
                    or message.additional_kwargs.get("reasoning_content")
                    or message.additional_kwargs.get("reasoning")
                )
                if reasoning is not None:
                    self.last_reasoning = reasoning


# ===========================================================================
# Agent Service
# ===========================================================================

class AgentService:
    """
    Manages LangGraph agent creation and execution.

    Uses the validated ``LLMConfig`` from ``llm_config_from_settings()`` or
    a user-supplied config dict for per-request overrides.

    **No fallbacks** — if the provider fails, the exception propagates so
    the API layer returns a deterministic error response.
    """

    def __init__(
        self,
        client: Optional[httpx.AsyncClient] = None,
        settings_obj: Optional[Settings] = None,
    ) -> None:
        self.client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(connect=15.0, read=None, write=30.0, pool=30.0)
        )
        self.settings = settings_obj or settings
        self.prompt_service = PromptService()
        self.default_system_prompt = self.prompt_service.get_system_prompt()
        self.tools = [web_search, document_search]

    @property
    def system_prompt(self) -> str:
        return self.default_system_prompt

    # ------------------------------------------------------------------
    # LLM instantiation
    # ------------------------------------------------------------------

    def _resolve_llm_config(
        self, llm_config: Optional[Dict[str, Any]] = None
    ) -> Union[OpenAIConfig, GeminiConfig, AnthropicConfig, OpenRouterConfig, CustomConfig]:
        """
        Build a typed ``LLMConfig`` from:
          1. A per-request ``llm_config`` dict (user-supplied model override), or
          2. The application environment settings.

        Raises ``DeepSeekAPIException`` with HTTP 400 if configuration is missing
        or invalid — fail-fast, no silent fallbacks.
        """
        effective = llm_config or get_active_llm_config()
        if effective and effective.get("api_key"):
            return self._build_config_from_dict(effective)

        # Fall back to env / settings
        try:
            return llm_config_from_settings(self.settings)
        except (ValueError, Exception) as exc:
            raise DeepSeekAPIException(
                message=f"LLM configuration error: {exc}. "
                        "Please configure your LLM provider in the environment or /llm-config.",
                status_code=400,
            ) from exc

    def _build_config_from_dict(
        self, d: Dict[str, Any]
    ) -> Union[OpenAIConfig, GeminiConfig, AnthropicConfig, OpenRouterConfig, CustomConfig]:
        """Build a typed LLMConfig from a user-supplied dict."""
        try:
            return llm_config_from_dict(d)
        except ValueError as exc:
            raise DeepSeekAPIException(message=str(exc), status_code=400) from exc

    def _get_llm_model(
        self,
        llm_config: Optional[Dict[str, Any]] = None,
        *,
        streaming: bool = False,
    ) -> Any:
        config = self._resolve_llm_config(llm_config)
        return build_chat_model(config, http_async_client=self.client, streaming=streaming)

    # ------------------------------------------------------------------
    # Agent executor builder
    # ------------------------------------------------------------------

    def _format_system_prompt(self, prompt: str) -> str:
        """Inject current date/time into system prompt."""
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
        behavior: Optional[str] = None,
    ) -> Any:
        """
        Build a LangGraph ReAct agent executor.

        Uses ``langgraph.prebuilt.create_react_agent`` (the recommended
        LangGraph-native approach) instead of the deprecated
        ``langchain.agents.create_agent``.
        """
        custom_behavior = system_prompt if system_prompt is not None else behavior
        model = self._get_llm_model(llm_config, streaming=True)
        prompt = self.prompt_service.get_system_prompt(behavior=custom_behavior)
        formatted_prompt = self._format_system_prompt(prompt)

        try:
            from langgraph.prebuilt import create_react_agent
            return create_react_agent(
                model=model,
                tools=self.tools,
                state_modifier=formatted_prompt,
            )
        except ImportError:
            # Fallback to langchain agent if langgraph not available
            try:
                from langchain.agents import create_agent
                return create_agent(
                    model=model,
                    tools=self.tools,
                    system_prompt=formatted_prompt,
                    debug=False,
                )
            except ImportError as exc:
                raise RuntimeError(
                    "Neither langgraph nor langchain.agents is available. "
                    "Install: pip install langgraph langchain"
                ) from exc

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
                kwargs: Dict[str, Any] = {}
                if reasoning_details is not None:
                    kwargs["reasoning_details"] = reasoning_details
                lc_messages.append(AIMessage(content=content, additional_kwargs=kwargs))
        return lc_messages

    # ------------------------------------------------------------------
    # Streaming chat response
    # ------------------------------------------------------------------

    async def astream_chat_response(
        self,
        messages: List[Dict[str, Any]],
        llm_config: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None,
    ):
        """
        Stream multi-event updates: tool status, reasoning deltas, text
        deltas, and a final ``done`` event.

        **Fail-fast**: On agent failure, yields a single ``error`` event
        and stops — no fallback content is generated.
        """
        agent_executor = self._create_agent_executor(
            llm_config=llm_config, system_prompt=system_prompt
        )
        lc_messages = self._convert_messages(messages)
        callback_handler = TokenAndReasoningCallbackHandler()

        full_content = ""
        full_reasoning = ""
        active_tool_depth = 0

        reset_search_counter()

        try:
            logger.info("[AgentService] Streaming agent execution…")
            async for event in agent_executor.astream_events(
                {"messages": lc_messages},
                version="v2",
                config={"callbacks": [callback_handler], "recursion_limit": 10},
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

                    if event_name == "web_search":
                        user_status = f"Searching web for '{query_str}'…" if query_str else "Searching web…"
                    elif event_name == "document_search":
                        user_status = f"Searching knowledge base for '{query_str}'…" if query_str else "Searching knowledge base…"
                    else:
                        user_status = f"Using {event_name}…"

                    yield {
                        "type": "status",
                        "status": "searching",
                        "tool": event_name,
                        "message": user_status,
                        "query": query_str,
                    }

                elif event_kind == "on_tool_end":
                    active_tool_depth = max(0, active_tool_depth - 1)
                    yield {
                        "type": "status",
                        "status": "tool_completed",
                        "tool": event_name,
                        "message": f"Finished {event_name}",
                    }

                elif event_kind == "on_chat_model_stream":
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

            # Estimate tokens if provider didn't report them
            if callback_handler.total_prompt_tokens == 0:
                callback_handler.total_prompt_tokens = max(
                    12, sum(len(str(m.get("content", ""))) // 4 for m in messages)
                )
            if callback_handler.total_completion_tokens == 0:
                callback_handler.total_completion_tokens = max(5, len(full_content) // 4)
            callback_handler.total_tokens = (
                callback_handler.total_prompt_tokens + callback_handler.total_completion_tokens
            )

            yield {
                "type": "done",
                "message": clean_thinking_text(full_content),
                "reasoning_details": callback_handler.last_reasoning or full_reasoning or None,
                "usage": {
                    "prompt_tokens": callback_handler.total_prompt_tokens,
                    "completion_tokens": callback_handler.total_completion_tokens,
                    "total_tokens": callback_handler.total_tokens,
                },
            }

        except Exception as exc:
            logger.exception("[AgentService] Agent execution failed: %s", exc)
            # NO fallback response — surface the real error
            err_str = str(exc)
            if "Recursion limit" in err_str or "GraphRecursionError" in type(exc).__name__:
                code = "RECURSION_LIMIT"
                message = "Agent search reached maximum recursion depth. Refine your query or reduce context."
            elif "RateLimit" in err_str or "429" in err_str:
                code = "RATE_LIMIT"
                message = "LLM provider rate limit reached. Please wait and retry."
            elif "AuthenticationError" in err_str or "401" in err_str:
                code = "AUTH_ERROR"
                message = "LLM API key is invalid or expired. Check your configuration."
            else:
                code = "AGENT_ERROR"
                message = f"Agent execution failed: {err_str}"

            yield {
                "type": "error",
                "code": code,
                "message": message,
            }

    # ------------------------------------------------------------------
    # Non-streaming chat response
    # ------------------------------------------------------------------

    async def get_chat_response(
        self,
        messages: List[Dict[str, Any]],
        llm_config: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Invoke agent and return final response dict.

        **Fail-fast**: On failure, raises — callers decide how to handle.
        """
        agent_executor = self._create_agent_executor(
            llm_config=llm_config, system_prompt=system_prompt
        )
        lc_messages = self._convert_messages(messages)
        callback_handler = TokenAndReasoningCallbackHandler()

        logger.info("[AgentService] Invoking agent…")
        response = await agent_executor.ainvoke(
            {"messages": lc_messages},
            config={"callbacks": [callback_handler], "recursion_limit": 10},
        )

        if isinstance(response, dict) and "messages" in response:
            content = response["messages"][-1].content
        else:
            content = getattr(response, "output", str(response))

        content = clean_thinking_text(content)

        if callback_handler.total_prompt_tokens == 0:
            callback_handler.total_prompt_tokens = max(
                12, sum(len(str(m.get("content", ""))) // 4 for m in messages)
            )
        if callback_handler.total_completion_tokens == 0:
            callback_handler.total_completion_tokens = max(5, len(content) // 4)
        callback_handler.total_tokens = (
            callback_handler.total_prompt_tokens + callback_handler.total_completion_tokens
        )

        return {
            "message": content,
            "usage": {
                "prompt_tokens": callback_handler.total_prompt_tokens,
                "completion_tokens": callback_handler.total_completion_tokens,
                "total_tokens": callback_handler.total_tokens,
            },
            "reasoning_details": callback_handler.last_reasoning,
        }


# ===========================================================================
# JSON repair utility
# ===========================================================================

def attempt_repair_json(content: str) -> Optional[Any]:
    """Attempt to repair truncated JSON strings."""
    if not content:
        return None

    start_brace = content.find("{")
    start_bracket = content.find("[")

    if start_brace == -1 and start_bracket == -1:
        return None

    start = (
        start_brace
        if (start_brace != -1 and (start_bracket == -1 or start_brace < start_bracket))
        else start_bracket
    )
    candidate = content[start:]

    last_brace = candidate.rfind("}")
    last_bracket = candidate.rfind("]")
    cut_off = max(last_brace, last_bracket)

    if cut_off != -1:
        truncated = candidate[: cut_off + 1]
        stack: List[str] = []
        in_string = False
        escape = False

        for char in truncated:
            if escape:
                escape = False
                continue
            if char == "\\":
                escape = True
                continue
            if char == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if char in "{[":
                stack.append(char)
            elif char == "}" and stack and stack[-1] == "{":
                stack.pop()
            elif char == "]" and stack and stack[-1] == "[":
                stack.pop()

        closing = "".join("}" if b == "{" else "]" for b in reversed(stack))
        try:
            return json.loads(truncated + closing, strict=False)
        except Exception:
            pass

    # Strategy 2: bracket-stack repair on full candidate
    stack = []
    in_string = False
    escape = False

    for char in candidate:
        if escape:
            escape = False
            continue
        if char == "\\":
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char in "{[":
            stack.append(char)
        elif char == "}" and stack and stack[-1] == "{":
            stack.pop()
        elif char == "]" and stack and stack[-1] == "[":
            stack.pop()

    suffix = '"' if in_string else ""
    suffix += "".join("}" if b == "{" else "]" for b in reversed(stack))

    try:
        return json.loads(candidate + suffix, strict=False)
    except Exception:
        return None


# ===========================================================================
# Schema normalisation & JSON parsing
# ===========================================================================

def _get_format_instructions(schema: Type[BaseModel]) -> str:
    """Return format instructions for the given schema."""
    if hasattr(schema, "format_instructions") and callable(getattr(schema, "format_instructions")):
        return schema.format_instructions()
    parser = PydanticOutputParser(pydantic_object=schema)
    return (
        parser.get_format_instructions()
        + "\nCRITICAL: Return ONLY a raw valid JSON object. No intro text, no preamble, no markdown."
    )


def _normalize_payload(
    parsed: Any,
    schema: Type[BaseModel],
    topic_hint: str = "",
) -> BaseModel:
    """Normalise raw parsed dict/list into target schema."""
    if hasattr(schema, "normalize") and callable(getattr(schema, "normalize")):
        return schema.normalize(parsed, hint=topic_hint)
    return schema.model_validate(parsed)


def preprocess_llm_json_string(content: str) -> str:
    """Fix invalid triple-quotes, unescaped code-span quotes, and block strings in LLM raw JSON responses."""
    if not content:
        return content

    def _fix_triple(match: re.Match) -> str:
        inner = match.group(1)
        inner_escaped = (
            inner.replace("\\", "\\\\")
            .replace('"', '\\"')
            .replace("\r\n", "\\n")
            .replace("\n", "\\n")
        )
        return f'"{inner_escaped}"'

    # Fix """...""" and '''...'''
    content = re.sub(r'"""([\s\S]*?)"""', _fix_triple, content)
    content = re.sub(r"'''([\s\S]*?)'''", _fix_triple, content)

    def _fix_backtick_quotes(match: re.Match) -> str:
        inner = match.group(0)
        # Replace unescaped " with ' inside backticks to prevent breaking JSON string delimiters
        return re.sub(r'(?<!\\)"', "'", inner)

    # Fix unescaped double quotes inside markdown code fences and backtick expressions
    content = re.sub(r'```[\s\S]*?```', _fix_backtick_quotes, content)
    content = re.sub(r'`[^`\n]+`', _fix_backtick_quotes, content)

    return content


def parse_json_from_llm_response(
    text: Any,
    schema: Type[BaseModel],
    topic_hint: str = "",
) -> BaseModel:
    """
    Extract and parse valid JSON from an LLM response, then validate against schema.

    Raises ``ValueError`` if no valid JSON can be extracted — no silent
    partial returns.
    """
    if not text:
        raise ValueError("Empty LLM response received")

    if isinstance(text, list):
        parts = []
        for block in text:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(block.get("text", "") or block.get("content", ""))
        text = "\n".join(parts)

    if not isinstance(text, str):
        text = str(text)

    logger.debug("[LLM Gateway] Raw response repr: %r", text[:3000])

    content = clean_thinking_text(text).strip()

    # Strip markdown code fences
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

    # Preprocess triple quotes and unescaped block strings
    content = preprocess_llm_json_string(content)

    # Direct parse
    try:
        parsed = json.loads(content, strict=False)
        return _normalize_payload(parsed, schema, topic_hint=topic_hint)
    except Exception as e:
        logger.debug("[LLM Gateway] Direct json.loads failed: %s", e)

    # Repair attempt
    repaired = attempt_repair_json(content)
    if repaired is not None:
        try:
            logger.info("[LLM Gateway] Using repaired JSON.")
            return _normalize_payload(repaired, schema, topic_hint=topic_hint)
        except Exception as e:
            logger.debug("[LLM Gateway] Repair validation failed: %s", e)

    # Substring extraction: first '{' to last '}'
    start = content.find("{")
    end = content.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(content[start : end + 1], strict=False)
            return _normalize_payload(parsed, schema, topic_hint=topic_hint)
        except Exception as e:
            logger.debug("[LLM Gateway] Brace-extraction failed: %s", e)

    # raw_decode from first '{'
    if start != -1:
        try:
            parsed, _ = json.JSONDecoder(strict=False).raw_decode(content[start:])
            return _normalize_payload(parsed, schema, topic_hint=topic_hint)
        except Exception:
            pass

    # Array fallback
    first_arr = content.find("[")
    last_arr = content.rfind("]")
    if first_arr != -1 and last_arr > first_arr:
        try:
            parsed = json.loads(content[first_arr : last_arr + 1], strict=False)
            return _normalize_payload(parsed, schema, topic_hint=topic_hint)
        except Exception:
            pass

    # Multi-object scan
    for match in re.finditer(r"\{", content):
        try:
            parsed, _ = json.JSONDecoder(strict=False).raw_decode(content[match.start() :])
            return _normalize_payload(parsed, schema, topic_hint=topic_hint)
        except Exception:
            continue

    raise ValueError(
        f"No valid JSON could be extracted from LLM response for schema "
        f"'{schema.__name__}'. Response (first 2000 chars): {text[:2000]}"
    )


# ===========================================================================
# Structured LLM gateway (non-streaming)
# ===========================================================================

async def call_llm_gateway(
    prompt: ChatPromptTemplate,
    params: Dict[str, Any],
    schema: Type[BaseModel],
) -> BaseModel:
    """
    Invoke the configured LLM provider and parse a structured Pydantic response.

    **No provider fallbacks** — uses the single configured provider from
    the environment. If the call fails, raises ``RuntimeError`` immediately.

    Parameters
    ----------
    prompt:
        LangChain ``ChatPromptTemplate`` with ``{format_instructions}`` variable.
    params:
        Template variables to inject into the prompt.
    schema:
        Pydantic model class to validate the response against.

    Returns
    -------
    BaseModel
        Validated instance of ``schema``.

    Raises
    ------
    RuntimeError
        If the LLM call fails or the response cannot be parsed.
    ValueError
        If settings are mis-configured.
    """
    fmt_inst = _get_format_instructions(schema)
    formatted_prompt = prompt.partial(format_instructions=fmt_inst)

    # Resolve config — prefer the logged-in user's saved config, else env
    try:
        user_config = get_active_llm_config()
        config = (
            llm_config_from_dict(user_config)
            if user_config and user_config.get("api_key")
            else llm_config_from_settings(settings)
        )
    except ValueError as exc:
        raise RuntimeError(f"LLM configuration error: {exc}") from exc

    # Respect configured max_tokens, capping at 4096 for structured JSON calls
    structured_max = min(config.max_tokens, 4096)

    # Build the model for this structured call
    # Use a fresh short-lived client (structured calls, not chat)
    llm = build_chat_model(
        type(config)(
            **{**config.model_dump(), "max_tokens": structured_max, "temperature": 0.2}
        ),
        streaming=False,
    )

    provider_name = config.provider
    topic_hint = str(params.get("concept_id", "") or params.get("topic", ""))

    logger.info(
        "[LLM Gateway] Calling '%s' | model='%s' | schema='%s' | max_tokens=%d",
        provider_name,
        config.model,
        schema.__name__,
        structured_max,
    )

    # Attempt native with_structured_output first if supported by model class
    try:
        if hasattr(llm, "with_structured_output"):
            structured_chain = formatted_prompt | llm.with_structured_output(schema)
            res = await structured_chain.ainvoke(params)
            if isinstance(res, schema):
                logger.info("[LLM Gateway] Native with_structured_output succeeded for %s", schema.__name__)
                return res
            if isinstance(res, dict):
                return _normalize_payload(res, schema, topic_hint=topic_hint)
    except Exception as exc:
        logger.debug("[LLM Gateway] Native with_structured_output fallback: %s", exc)

    try:
        rendered_msgs = formatted_prompt.format_prompt(**params).to_messages()
        rendered_text = "\n---\n".join(f"[{m.type.upper()}]: {m.content}" for m in rendered_msgs)
    except Exception:
        rendered_text = str(params)

    logger.info("[LLM Gateway] Prompt (schema=%s):\n%s", schema.__name__, rendered_text[:1000])

    chain = formatted_prompt | llm
    t0 = time.monotonic()
    try:
        res = await chain.ainvoke(params)
    except Exception as exc:
        raise RuntimeError(
            f"LLM provider '{provider_name}' failed for schema '{schema.__name__}': {exc}"
        ) from exc

    elapsed_ms = int((time.monotonic() - t0) * 1000)
    content = res.content if hasattr(res, "content") else str(res)

    if isinstance(content, list):
        content = "\n".join(
            b if isinstance(b, str) else b.get("text", "") or b.get("content", "")
            for b in content
        )
    elif not isinstance(content, str):
        content = str(content)

    logger.info(
        "[LLM Gateway] ⏱ %dms | schema='%s' | chars=%d",
        elapsed_ms,
        schema.__name__,
        len(content),
    )
    logger.debug("[LLM Gateway] Raw response:\n%s", content[:1500])

    try:
        return parse_json_from_llm_response(content, schema, topic_hint=topic_hint)
    except ValueError as exc:
        raise RuntimeError(
            f"LLM response for schema '{schema.__name__}' could not be parsed: {exc}"
        ) from exc


# ===========================================================================
# Streaming text gateway
# ===========================================================================

async def call_llm_gateway_stream(
    prompt: ChatPromptTemplate,
    params: Dict[str, Any],
    schema: Optional[Type[BaseModel]] = None,
):
    """
    Async generator yielding text token delta strings from the configured LLM.

    Parameters
    ----------
    prompt : ChatPromptTemplate
        Prompt to send to LLM.
    params : Dict[str, Any]
        Variables to format into the prompt.
    schema : Optional[Type[BaseModel]], default None
        Optional Pydantic schema for structured generation. If None, performs
        pure text streaming (Markdown / conversational text) without whole-document schema.

    **No provider fallbacks** — uses the single configured provider.

    Raises
    ------
    RuntimeError
        If the LLM call fails.
    """
    if schema is not None:
        fmt_inst = _get_format_instructions(schema)
        if "format_instructions" in prompt.input_variables:
            formatted_prompt = prompt.partial(format_instructions=fmt_inst)
        else:
            formatted_prompt = prompt
    else:
        if "format_instructions" in prompt.input_variables:
            formatted_prompt = prompt.partial(format_instructions="")
        else:
            formatted_prompt = prompt


    try:
        user_config = get_active_llm_config()
        config = (
            llm_config_from_dict(user_config)
            if user_config and user_config.get("api_key")
            else llm_config_from_settings(settings)
        )
    except ValueError as exc:
        raise RuntimeError(f"LLM configuration error: {exc}") from exc

    safe_max = min(config.max_tokens, 4096)

    llm = build_chat_model(
        type(config)(
            **{**config.model_dump(), "max_tokens": safe_max, "temperature": 0.2}
        ),
        streaming=True,
    )

    logger.info(
        "[LLM Gateway Stream] Starting stream | provider='%s' | model='%s'",
        config.provider,
        config.model,
    )

    chain = formatted_prompt | llm
    chunk_count = 0
    try:
        async for chunk in chain.astream(params):
            norm = StreamEventNormalizer.normalize_chunk(chunk)
            if norm.text:
                chunk_count += 1
                yield norm.text
    except Exception as exc:
        raise RuntimeError(
            f"LLM streaming provider '{config.provider}' failed: {exc}"
        ) from exc

    if chunk_count == 0:
        raise RuntimeError(
            f"LLM provider '{config.provider}' returned no streaming content."
        )