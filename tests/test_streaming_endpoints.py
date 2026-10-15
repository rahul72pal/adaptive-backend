import json
import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import patch, AsyncMock
from main import app
from app.core.dependencies import get_current_user
from app.db.models import User
import uuid

mock_user = User(
    id=uuid.uuid4(),
    email="stream_test@example.com",
    name="Stream Tester",
    hashed_password="mock_hash"
)

@pytest.fixture
def mock_auth():
    app.dependency_overrides[get_current_user] = lambda: mock_user
    yield
    app.dependency_overrides.pop(get_current_user, None)


@pytest.mark.asyncio
async def test_lesson_streaming_protocol_endpoint(mock_auth):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        # Mock LLM stream & parsing
        mock_lesson_content = {
            "concept_id": "array_two_pointer",
            "intuition": "Pointers converge inward from ends.",
            "core_idea": "Use two index markers to scan sorted collection.",
            "mechanism": "1. Initialize left and right.\n2. Adjust indices.",
            "step_by_step_example": "Trace on [2, 7, 11, 15] target=9.",
            "code_snippet": "def two_sum(): pass",
            "language": "python",
            "complexity": {
                "best_case": "O(1)",
                "worst_case": "O(n)",
                "space": "O(1)",
                "explanation": "Single linear pass"
            },
            "edge_cases": ["empty array"],
            "common_mistakes": ["forgetting to sort"],
            "checkpoint": {
                "question": "What happens if array is not sorted?",
                "options": ["Algorithm fails", "Runs in O(n^2)", "Throws error", "Always works"],
                "correct_option": "Algorithm fails",
                "explanation": "Two-pointer invariant requires monotonicity."
            }
        }

        async def mock_stream_chunks(*args, **kwargs):
            yield json.dumps(mock_lesson_content)

        with patch("app.llm.gateway.call_llm_gateway_stream", side_effect=mock_stream_chunks):
            res = await client.post(
                "/api/learning/lesson/stream",
                json={
                    "topic": "Array",
                    "concept_id": "array_two_pointer",
                    "level": "beginner",
                    "pattern": "two_pointer"
                }
            )
            assert res.status_code == 200
            assert "text/event-stream" in res.headers["content-type"]

            # Parse SSE events from response body
            body_text = res.text
            events = []
            for block in body_text.split("\n\n"):
                if block.strip().startswith("data: "):
                    events.append(json.loads(block.strip()[6:]))

            event_types = [e.get("event") for e in events]

            # Verify standard events were emitted
            assert "message_start" in event_types
            assert "text_delta" in event_types
            assert "user_input" in event_types
            assert "message_end" in event_types

            # Verify user_input event structure
            user_input_evt = next(e for e in events if e.get("event") == "user_input")
            input_def = user_input_evt["input"]
            assert input_def["input_id"] == "checkpoint_array_two_pointer"
            assert input_def["type"] == "single_select"
            assert len(input_def["options"]) >= 4
            assert input_def["required"] is True

            # Verify message_end status is 'awaiting_user'
            message_end_evt = next(e for e in events if e.get("event") == "message_end")
            assert message_end_evt["status"] == "awaiting_user"
            assert "payload" in message_end_evt


@pytest.mark.asyncio
async def test_checkpoint_answer_endpoint(mock_auth):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        res = await client.post(
            "/api/learning/lesson/checkpoint-answer",
            json={
                "topic": "Array",
                "concept_id": "array_two_pointer",
                "input_id": "checkpoint_array_two_pointer",
                "answer": "Algorithm fails"
            }
        )
        assert res.status_code == 200
        data = res.json()
        assert "is_correct" in data
        assert "feedback" in data
        assert data["input_id"] == "checkpoint_array_two_pointer"
