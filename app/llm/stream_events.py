"""
Structured LLM Streaming Protocol & Event Models (Pydantic v2)
==============================================================

Defines a consistent, typed event protocol for all Server-Sent Events (SSE)
streaming responses across diagnostic chat and lesson generation.

Supported event types:
  - message_start
  - text_delta
  - user_input
  - tool_start
  - tool_end
  - error
  - message_end
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field


# ===========================================================================
# 1. User Input Components & Validation Models
# ===========================================================================

InputType = Literal["single_select", "multi_select", "text", "number", "form"]
StreamStatus = Literal["completed", "awaiting_user", "error", "generating"]


class InputOption(BaseModel):
    """Selectable option for single_select and multi_select input types."""
    id: str = Field(..., description="Unique option identifier (e.g., 'A', 'opt_1')")
    label: str = Field(..., description="Display label for the user")
    value: Optional[Any] = Field(default=None, description="Submitted value or payload")
    description: Optional[str] = Field(default=None, description="Optional extra detail")


class ValidationConstraints(BaseModel):
    """Validation rules tailored to input type."""
    min: Optional[Union[int, float]] = None
    max: Optional[Union[int, float]] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    pattern: Optional[str] = None
    options: Optional[List[str]] = None
    fields: Optional[Dict[str, Any]] = None  # Sub-schema for form inputs


class UserInputPayload(BaseModel):
    """Standardized interactive input schema requested from the client."""
    input_id: str = Field(..., description="Unique identifier for this input request")
    type: InputType = Field(..., description="Input component type")
    label: str = Field(..., description="Main prompt or question label")
    description: Optional[str] = Field(default=None, description="Help text or context")
    options: Optional[List[InputOption]] = Field(default=None, description="Options for select types")
    required: bool = Field(default=True, description="Whether an answer is required")
    default_value: Optional[Any] = Field(default=None, description="Default or initial value")
    validation: Optional[ValidationConstraints] = Field(default=None, description="Validation rules")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Domain metadata (e.g., concept_id, hints)")


class UserInputSubmission(BaseModel):
    """Payload sent by client when answering a user_input request."""
    input_id: str = Field(..., description="Matches the requested input_id")
    value: Any = Field(..., description="User's selected option, text, number, or form dict")


def validate_user_input_submission(schema: UserInputPayload, value: Any) -> tuple[bool, Optional[str]]:
    """
    Validates submitted user input against the original UserInputPayload schema.
    Returns (is_valid, error_message).
    """
    if schema.required and (value is None or (isinstance(value, str) and not value.strip())):
        return False, f"Field '{schema.label}' is required."

    if value is None and not schema.required:
        return True, None

    v = schema.validation

    if schema.type == "single_select":
        allowed_ids = {opt.id for opt in (schema.options or [])}
        allowed_labels = {opt.label for opt in (schema.options or [])}
        val_str = str(value).strip()
        if val_str not in allowed_ids and val_str not in allowed_labels:
            if v and v.options and val_str not in v.options:
                return False, f"Invalid selection '{value}'. Must be one of {[opt.label for opt in (schema.options or [])]}."

    elif schema.type == "multi_select":
        if not isinstance(value, (list, set, tuple)):
            return False, "Multi-select input must be a list of selections."
        allowed_ids = {opt.id for opt in (schema.options or [])}
        allowed_labels = {opt.label for opt in (schema.options or [])}
        for item in value:
            item_str = str(item).strip()
            if item_str not in allowed_ids and item_str not in allowed_labels:
                return False, f"Invalid selection '{item}' in multi-select."
        if v:
            if v.min_length and len(value) < v.min_length:
                return False, f"At least {v.min_length} selections required."
            if v.max_length and len(value) > v.max_length:
                return False, f"At most {v.max_length} selections allowed."

    elif schema.type == "text":
        val_str = str(value)
        if v:
            if v.min_length and len(val_str) < v.min_length:
                return False, f"Text must be at least {v.min_length} characters."
            if v.max_length and len(val_str) > v.max_length:
                return False, f"Text must be at most {v.max_length} characters."
            if v.pattern:
                import re
                if not re.search(v.pattern, val_str):
                    return False, f"Text does not match required pattern."

    elif schema.type == "number":
        try:
            num = float(value)
            if v:
                if v.min is not None and num < v.min:
                    return False, f"Value must be at least {v.min}."
                if v.max is not None and num > v.max:
                    return False, f"Value must be at most {v.max}."
        except (ValueError, TypeError):
            return False, f"Value '{value}' is not a valid number."

    elif schema.type == "form":
        if not isinstance(value, dict):
            return False, "Form input must be a JSON object / dict."

    return True, None


# ===========================================================================
# 2. Typed Stream Event Envelope & Payloads
# ===========================================================================

def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id() -> str:
    return str(uuid.uuid4())


class MessageStartEvent(BaseModel):
    """Signals initialization of a streaming message."""
    event: Literal["message_start"] = "message_start"
    event_id: str = Field(default_factory=_new_id)
    timestamp: str = Field(default_factory=_utcnow_iso)
    session_id: Optional[str] = None
    conversation_id: Optional[str] = None
    role: str = "assistant"
    metadata: Optional[Dict[str, Any]] = None


class TextDeltaEvent(BaseModel):
    """Incremental Markdown or plain-text token chunk."""
    event: Literal["text_delta"] = "text_delta"
    event_id: str = Field(default_factory=_new_id)
    timestamp: str = Field(default_factory=_utcnow_iso)
    delta: str = Field(..., description="Text chunk to append")
    index: Optional[int] = None


class UserInputEvent(BaseModel):
    """Emitted when agent/engine requires structured user input and pauses."""
    event: Literal["user_input"] = "user_input"
    event_id: str = Field(default_factory=_new_id)
    timestamp: str = Field(default_factory=_utcnow_iso)
    input: UserInputPayload = Field(..., description="Structured input definition")


class ToolStartEvent(BaseModel):
    """Signals tool execution initiation."""
    event: Literal["tool_start"] = "tool_start"
    event_id: str = Field(default_factory=_new_id)
    timestamp: str = Field(default_factory=_utcnow_iso)
    tool_name: str
    tool_input: Optional[Any] = None


class ToolEndEvent(BaseModel):
    """Signals tool execution completion."""
    event: Literal["tool_end"] = "tool_end"
    event_id: str = Field(default_factory=_new_id)
    timestamp: str = Field(default_factory=_utcnow_iso)
    tool_name: str
    tool_output: Optional[Any] = None


class ErrorEvent(BaseModel):
    """Communicates stream failure safely without leaking sensitive traces."""
    event: Literal["error"] = "error"
    event_id: str = Field(default_factory=_new_id)
    timestamp: str = Field(default_factory=_utcnow_iso)
    message: str = Field(..., description="Sanitized error description")
    code: Optional[str] = None


class MessageEndEvent(BaseModel):
    """Terminal event marking completion or pause for user input."""
    event: Literal["message_end"] = "message_end"
    event_id: str = Field(default_factory=_new_id)
    timestamp: str = Field(default_factory=_utcnow_iso)
    status: StreamStatus = Field(default="completed", description="'completed' or 'awaiting_user'")
    session_id: Optional[str] = None
    total_tokens: Optional[int] = None
    payload: Optional[Dict[str, Any]] = Field(default=None, description="Final payload or state snapshot")


# Discriminated Union
StreamEvent = Union[
    MessageStartEvent,
    TextDeltaEvent,
    UserInputEvent,
    ToolStartEvent,
    ToolEndEvent,
    ErrorEvent,
    MessageEndEvent,
]


# ===========================================================================
# 3. SSE Serializer
# ===========================================================================

def format_sse(event_data: Union[BaseModel, Dict[str, Any]]) -> str:
    """Encodes a typed event or dict into a valid Server-Sent Events frame."""
    if isinstance(event_data, BaseModel):
        payload = event_data.model_dump(exclude_none=True)
    else:
        payload = {k: v for k, v in event_data.items() if v is not None}
    return f"data: {json.dumps(payload)}\n\n"
