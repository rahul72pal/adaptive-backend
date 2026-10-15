import json
import pytest
from app.llm.stream_events import (
    MessageStartEvent,
    TextDeltaEvent,
    UserInputEvent,
    UserInputPayload,
    InputOption,
    ValidationConstraints,
    MessageEndEvent,
    ErrorEvent,
    ToolStartEvent,
    ToolEndEvent,
    format_sse,
    validate_user_input_submission,
)


def test_message_start_event():
    evt = MessageStartEvent(session_id="sess_123", role="assistant", metadata={"turn": 1})
    sse = format_sse(evt)
    assert sse.startswith("data: ")
    data = json.loads(sse[6:].strip())
    assert data["event"] == "message_start"
    assert data["session_id"] == "sess_123"
    assert data["role"] == "assistant"
    assert data["metadata"]["turn"] == 1
    assert "timestamp" in data
    assert "event_id" in data


def test_text_delta_event():
    evt = TextDeltaEvent(delta="### 💡 Concept Probe\n\nExplain binary search mechanism.")
    sse = format_sse(evt)
    data = json.loads(sse[6:].strip())
    assert data["event"] == "text_delta"
    assert "### 💡 Concept Probe" in data["delta"]


def test_user_input_event_single_select():
    options = [
        InputOption(id="A", label="O(1) auxiliary space", value="O(1)"),
        InputOption(id="B", label="O(n) auxiliary space", value="O(n)"),
    ]
    u_input = UserInputPayload(
        input_id="chk_binary_search",
        type="single_select",
        label="Space Complexity Checkpoint",
        description="What is the space complexity of iterative binary search?",
        options=options,
        required=True,
        validation=ValidationConstraints(options=["O(1)", "O(n)", "A", "B", "O(1) auxiliary space", "O(n) auxiliary space"]),
        metadata={"correct_option": "A"}
    )
    evt = UserInputEvent(input=u_input)
    sse = format_sse(evt)
    data = json.loads(sse[6:].strip())
    assert data["event"] == "user_input"
    assert data["input"]["input_id"] == "chk_binary_search"
    assert data["input"]["type"] == "single_select"
    assert len(data["input"]["options"]) == 2

    # Validation tests
    valid, err = validate_user_input_submission(u_input, "A")
    assert valid is True
    assert err is None

    valid, err = validate_user_input_submission(u_input, "O(1) auxiliary space")
    assert valid is True

    valid, err = validate_user_input_submission(u_input, "Invalid Option")
    assert valid is False
    assert err is not None

    valid, err = validate_user_input_submission(u_input, "")
    assert valid is False


def test_user_input_event_text():
    u_input = UserInputPayload(
        input_id="diag_input_001",
        type="text",
        label="Diagnostic Explanation",
        description="Explain two pointer convergence.",
        required=True,
        validation=ValidationConstraints(min_length=5, max_length=500)
    )
    valid, err = validate_user_input_submission(u_input, "Pointers start at left and right indices.")
    assert valid is True

    valid, err = validate_user_input_submission(u_input, "hi")
    assert valid is False
    assert "at least 5 characters" in err


def test_message_end_event_awaiting_user():
    evt = MessageEndEvent(
        status="awaiting_user",
        session_id="sess_456",
        payload={"turn": 2, "is_complete": False}
    )
    sse = format_sse(evt)
    data = json.loads(sse[6:].strip())
    assert data["event"] == "message_end"
    assert data["status"] == "awaiting_user"
    assert data["session_id"] == "sess_456"
    assert data["payload"]["is_complete"] is False


def test_tool_start_and_end_events():
    t_start = ToolStartEvent(tool_name="curriculum_engine", tool_input={"concept": "bfs"})
    t_end = ToolEndEvent(tool_name="curriculum_engine", tool_output={"status": "success"})

    sse_start = format_sse(t_start)
    sse_end = format_sse(t_end)

    d_start = json.loads(sse_start[6:].strip())
    d_end = json.loads(sse_end[6:].strip())

    assert d_start["event"] == "tool_start"
    assert d_start["tool_name"] == "curriculum_engine"
    assert d_end["event"] == "tool_end"
    assert d_end["tool_output"]["status"] == "success"


def test_error_event():
    evt = ErrorEvent(message="LLM provider timed out.", code="TIMEOUT_ERROR")
    sse = format_sse(evt)
    data = json.loads(sse[6:].strip())
    assert data["event"] == "error"
    assert data["code"] == "TIMEOUT_ERROR"
    assert data["message"] == "LLM provider timed out."
