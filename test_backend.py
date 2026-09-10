import asyncio
import uuid
import pytest
from httpx import AsyncClient, ASGITransport
from main import app

@pytest.mark.asyncio
async def test_full_platform_flow():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        # 1. Health Check
        res = await client.get("/health")
        assert res.status_code == 200
        assert res.json()["status"] == "healthy"

        # 2. Register User
        user_email = f"testuser_{uuid.uuid4().hex[:6]}@example.com"
        reg_payload = {
            "email": user_email,
            "name": "Test Learner",
            "password": "SecretPassword123!"
        }
        res = await client.post("/api/auth/register", json=reg_payload)
        assert res.status_code == 201
        data = res.json()
        assert "access_token" in data
        token = data["access_token"]
        user_id = data["user"]["id"]
        headers = {"Authorization": f"Bearer {token}"}

        # 3. Authenticate / Login check
        res = await client.get("/api/auth/me", headers=headers)
        assert res.status_code == 200
        assert res.json()["email"] == user_email

        # 4. Start Diagnostic Chat Session
        diag_start_payload = {"topic": "React"}
        res = await client.post("/api/diagnostic/start", json=diag_start_payload, headers=headers)
        assert res.status_code == 201
        diag_data = res.json()
        assert "session_id" in diag_data
        session_id = diag_data["session_id"]
        assert "current_question" in diag_data

        # 5. Answer Diagnostic Turn
        ans_payload = {
            "session_id": session_id,
            "answer": "I am deeply experienced with state and props in React."
        }
        res = await client.post("/api/diagnostic/answer", json=ans_payload, headers=headers)
        assert res.status_code == 200
        ans_data = res.json()
        assert "knowledge_profile" in ans_data

        # 6. Opt-Out Diagnostic (Complete Beginner Test)
        opt_payload = {"session_id": session_id}
        res = await client.post("/api/diagnostic/opt-out", json=opt_payload, headers=headers)
        assert res.status_code == 200
        assert res.json()["is_complete"] is True

        # 7. Generate Personalized Learning Path
        path_payload = {
            "topic": "React",
            "target_level": "expert",
            "diagnostic_session_id": session_id
        }
        res = await client.post("/api/learning/generate-path", json=path_payload, headers=headers)
        assert res.status_code == 201
        lp_data = res.json()
        assert "nodes" in lp_data
        assert len(lp_data["nodes"]) > 0

        # 8. Retrieve Learning Path
        res = await client.get(f"/api/learning/{user_id}/react", headers=headers)
        assert res.status_code == 200
        assert res.json()["user_id"] == user_id

        # 9. Generate Question for concept 'state'
        q_gen_payload = {
            "topic": "React",
            "concept_id": "state",
            "concept_type": "code",
            "difficulty": 2,
            "question_type": "mcq"
        }
        res = await client.post("/api/questions/generate", json=q_gen_payload, headers=headers)
        assert res.status_code == 200
        q_data = res.json()
        assert "question" in q_data
        q_id = q_data["_id"]

        # 10. Submit Question Attempt
        attempt_payload = {
            "question_id": q_id,
            "concept_id": "state",
            "topic": "react",
            "difficulty": 2,
            "answer": "const [count, setCount] = useState(0);"
        }
        res = await client.post("/api/attempts/submit", json=attempt_payload, headers=headers)
        assert res.status_code == 200
        att_data = res.json()
        assert "score" in att_data
        assert "mastery_update" in att_data

        # 11. Retrieve Deterministic Progress & Mastery Breakdown
        res = await client.get(f"/api/progress/{user_id}/react", headers=headers)
        assert res.status_code == 200
        prog_data = res.json()
        assert prog_data["topic"] == "react"
        assert "overall_mastery" in prog_data
        assert "current_level" in prog_data

        # 12. Test Adaptive Re-Teaching Node
        reteach_payload = {
            "topic": "React",
            "concept_id": "state",
            "strategies_tried": ["analogy"],
            "consecutive_failures": 1
        }
        res = await client.post("/api/learning/reteach", json=reteach_payload, headers=headers)
        assert res.status_code == 200
        reteach_data = res.json()
        assert reteach_data["current_strategy"] == "visual_diagram"

        print("\nAll Backend End-to-End Verification Tests Passed Successfully!")

if __name__ == "__main__":
    asyncio.run(test_full_platform_flow())
