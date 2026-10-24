import pytest
from httpx import AsyncClient, ASGITransport
import uuid

from main import app
from app.catalog.models import TopicMetadata, CategoryMetadata
from app.services.topic_catalog_service import get_catalog_service
from app.core.dependencies import get_current_user
from app.db.models import User
from app.engine.curriculum_engine import (
    validate_concept_nodes,
    validate_topological_sort,
    get_available_nodes,
    slugify_topic,
    normalize_topic,
)
from app.llm.schemas import (
    CheckpointQuestion,
    GeneratedQuestionItem,
    EvaluatorGradeSchema,
    ScoreEvaluationSchema,
)
from app.graphs.reteach_graph import select_strategy_node

mock_audit_user = User(
    id=uuid.uuid4(),
    email="audit_user@example.com",
    name="System Auditor",
    hashed_password="mock_hash_audit"
)

@pytest.fixture
def mock_auth():
    app.dependency_overrides[get_current_user] = lambda: mock_audit_user
    yield
    app.dependency_overrides.pop(get_current_user, None)


# =============================================================================
# 1. API SECURITY & AUTH ENFORCEMENT
# =============================================================================

@pytest.mark.asyncio
async def test_auth_rejection_for_unprotected_calls():
    transport = ASGITransport(app=app)
    # Note: NO mock_auth fixture active here -> testing unauthenticated access
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        # 1. Select topic without auth -> 401
        res_sel = await client.post("/api/topics/select", json={"topic": "t_08_binary_search"})
        assert res_sel.status_code == 401

        # 2. Topic mode choice without auth -> 401
        res_mode = await client.post("/api/topics/mode", json={"topic": "t_08_binary_search", "mode_choice": "diagnostic"})
        assert res_mode.status_code == 401

        # 3. Diagnostic start without auth -> 401
        res_diag = await client.post("/api/diagnostic/start", json={"topic": "t_08_binary_search"})
        assert res_diag.status_code == 401

        # 4. Generate learning path without auth -> 401
        res_path = await client.post("/api/learning/generate-path", json={"topic": "t_08_binary_search", "target_level": "intermediate"})
        assert res_path.status_code == 401

        # 5. Complete node without auth -> 401
        res_comp = await client.post("/api/learning/complete-node", json={"topic": "t_08_binary_search", "concept_id": "bs_invariants"})
        assert res_comp.status_code == 401


# =============================================================================
# 2. CATALOG INTEGRITY & EDGE CASE RESOLUTION
# =============================================================================

@pytest.mark.asyncio
async def test_catalog_query_validation_bounds():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        # Invalid page (page < 1) -> 422
        res_page = await client.get("/api/topics/catalog?page=0")
        assert res_page.status_code == 422

        # Invalid page_size (page_size > 2500) -> 422
        res_size = await client.get("/api/topics/catalog?page_size=3000")
        assert res_size.status_code == 422

        # Non-existent topic_id -> 404
        res_not_found = await client.get("/api/topics/catalog/t_99_non_existent_topic_id")
        assert res_not_found.status_code == 404

        # Ambiguous topic without preferred_category_id -> returns candidates with is_ambiguous=True
        res_ambig = await client.get("/api/topics/resolve?topic=Binary+Search")
        assert res_ambig.status_code == 200
        ambig_data = res_ambig.json()
        assert ambig_data["status"] == "ambiguous"
        assert ambig_data["result"]["is_resolved"] is False
        assert ambig_data["result"]["is_ambiguous"] is True
        assert len(ambig_data["result"]["matching_candidates"]) >= 2

        # Disambiguated topic with category_id -> resolves cleanly
        res_disambig = await client.get("/api/topics/resolve?topic=Binary+Search&category_id=cat_08")
        assert res_disambig.status_code == 200
        disambig_data = res_disambig.json()
        assert disambig_data["status"] == "resolved"
        assert disambig_data["result"]["is_resolved"] is True
        assert disambig_data["result"]["topic"]["topic_id"] == "t_08_binary_search"


# =============================================================================
# 3. ADVERSARIAL GRAPH VALIDATION
# =============================================================================

def test_graph_validation_missing_fields_and_invalid_types():
    # Empty concept_id -> ValueError
    bad_nodes_missing = [
        {"concept_id": "", "level": "beginner"}
    ]
    with pytest.raises(ValueError) as excinfo:
        validate_concept_nodes(bad_nodes_missing, level="beginner")
    assert "missing or empty concept_id" in str(excinfo.value).lower()

    # Duplicate concept_id -> ValueError
    duplicate_nodes = [
        {"concept_id": "c_dup", "level": "beginner", "node_type": "learning", "depends_on": []},
        {"concept_id": "c_dup", "level": "beginner", "node_type": "practice", "depends_on": []}
    ]
    with pytest.raises(ValueError) as excinfo_dup:
        validate_concept_nodes(duplicate_nodes, level="beginner")
    assert "duplicate concept_id" in str(excinfo_dup.value).lower()


def test_graph_validation_self_loop_pruning():
    nodes_with_self_loop = [
        {"concept_id": "c_self", "level": "beginner", "node_type": "learning", "depends_on": ["c_self"]}
    ]
    validate_concept_nodes(nodes_with_self_loop, level="beginner")
    assert nodes_with_self_loop[0]["depends_on"] == [], "Self-dependency was not pruned!"


def test_graph_topological_sort_cycle_breaking():
    # Adversarial cycle: A -> B and B -> A
    cyclic_nodes = [
        {"concept_id": "node_a", "id": "node_a", "depends_on": ["node_b"]},
        {"concept_id": "node_b", "id": "node_b", "depends_on": ["node_a"]}
    ]
    # Kahn's algorithm cycle resolution should break cycles gracefully without hanging
    resolved = validate_topological_sort(cyclic_nodes)
    assert len(resolved) == 2
    # At least one dependency must be pruned to eliminate the cycle
    assert resolved[0]["depends_on"] == [] or resolved[1]["depends_on"] == []


def test_get_available_nodes_dag_progression():
    nodes = [
        {"concept_id": "root_1", "status": "learning", "depends_on": []},
        {"concept_id": "child_1", "status": "learning", "depends_on": ["root_1"]},
        {"concept_id": "child_2", "status": "learning", "depends_on": ["child_1"]}
    ]

    # Initially, only root_1 is available
    avail_0 = get_available_nodes(nodes, completed_ids=set())
    assert [n["concept_id"] for n in avail_0] == ["root_1"]

    # When root_1 is mastered, its status updates and child_1 unlocks
    nodes[0]["status"] = "mastered"
    avail_1 = get_available_nodes(nodes, completed_ids={"root_1"})
    assert [n["concept_id"] for n in avail_1] == ["child_1"]

    # When child_1 is mastered, its status updates and child_2 unlocks
    nodes[1]["status"] = "mastered"
    avail_2 = get_available_nodes(nodes, completed_ids={"root_1", "child_1"})
    assert [n["concept_id"] for n in avail_2] == ["child_2"]


# =============================================================================
# 4. RE-TEACHING PEDAGOGICAL STRATEGY PROGRESSION
# =============================================================================

@pytest.mark.asyncio
async def test_reteach_strategy_transition_cycle():
    from unittest.mock import patch
    from app.llm.schemas import LessonExplanationSchema

    async def fake_reteach_gateway(prompt, params, schema):
        return LessonExplanationSchema(
            concept_id=params.get("concept_id", "concept"),
            strategy=params.get("strategy", "analogy"),
            content="Tailored pedagogical explanation focusing on mental models."
        )

    with patch("app.graphs.reteach_graph.call_llm_gateway", side_effect=fake_reteach_gateway):
        base_state = {"topic": "t_08_binary_search", "concept_id": "bs_loop_invariants"}

        # Attempt 1: Default start is analogy
        s1 = await select_strategy_node({**base_state, "strategies_tried": [], "consecutive_failures": 1})
        assert s1["current_strategy"] == "analogy"

        # Attempt 2: After analogy, transitions to visual_diagram
        s2 = await select_strategy_node({**base_state, "strategies_tried": ["analogy"], "consecutive_failures": 2})
        assert s2["current_strategy"] == "visual_diagram"

        # Attempt 3: After visual_diagram, transitions to worked_example
        s3 = await select_strategy_node({**base_state, "strategies_tried": ["analogy", "visual_diagram"], "consecutive_failures": 2})
        assert s3["current_strategy"] == "worked_example"

        # Threshold of >= 3 consecutive failures triggers prerequisite remediation
        s_fallback = await select_strategy_node({**base_state, "strategies_tried": ["analogy", "visual_diagram"], "consecutive_failures": 3})
        assert s_fallback["action"] == "spot_check_prereq"



# =============================================================================
# 5. ASSESSMENT MCQ & TEST-CASE VALIDATION
# =============================================================================

def test_mcq_checkpoint_validation():
    # Valid 4-option checkpoint
    cp = CheckpointQuestion(
        question="What is the space complexity of an in-place array reversal?",
        options=["O(1)", "O(log n)", "O(n)", "O(n^2)"],
        correct_option="O(1)",
        explanation="In-place algorithms modify elements within the existing array."
    )
    assert cp.correct_option == "O(1)"
    assert len(cp.options) == 4

    # Invalid options count (<4) raises ValidationError
    with pytest.raises(Exception):
        CheckpointQuestion(
            question="What is the space complexity?",
            options=["O(1)", "O(n)"],
            correct_option="O(1)",
            explanation="Too few options"
        )


def test_generated_question_schema_mcq_options():
    q = GeneratedQuestionItem(
        type="mcq",
        concept_id="array_two_pointer",
        difficulty=3,
        question="What is the runtime of two pointers on a sorted array of length n?",
        options=["O(n)", "O(n log n)", "O(n^2)", "O(1)"],
        correct_option="O(n)",
        explanation="Each element is visited at most once."
    )
    assert q.type == "mcq"
    assert q.correct_option in q.options
    assert q.difficulty == 3


# =============================================================================
# 6. API ERROR HANDLING & EDGE CASES
# =============================================================================

@pytest.mark.asyncio
async def test_api_missing_records_graceful_failures(mock_auth):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        # 1. Answer non-existent diagnostic session -> 404
        bad_session_id = str(uuid.uuid4())
        res_diag_bad = await client.post(
            "/api/diagnostic/answer",
            json={"session_id": bad_session_id, "answer": "Sample answer."}
        )
        assert res_diag_bad.status_code == 404

        # 2. Complete level with non-existent learning_path_id -> 404
        res_lvl_bad = await client.post(
            "/api/learning/complete-level",
            json={"learning_path_id": "lp_non_existent_id_999"}
        )
        assert res_lvl_bad.status_code == 404

        # 3. Health check -> 200 Healthy
        res_health = await client.get("/health")
        assert res_health.status_code == 200
        assert res_health.json()["status"] == "healthy"


@pytest.mark.asyncio
async def test_topic_mode_disambiguation_and_auto_resolution(mock_auth):
    """
    Validates that:
    1. Setting mode using stable topic_id resolves immediately.
    2. Setting mode with ambiguous topic name and category_id disambiguates cleanly.
    3. Setting mode with ambiguous topic name auto-resolves when user has that topic selected.
    """
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        # 1. Select the specific topic
        res_sel = await client.post(
            "/api/topics/select",
            json={"topic": "t_16_recursion_fundamentals", "category_id": "cat_16_recursion"}
        )
        assert res_sel.status_code == 200
        assert res_sel.json()["topic_id"] == "t_16_recursion_fundamentals"

        # 2. Set mode by exact topic_id
        res_mode_id = await client.post(
            "/api/topics/mode",
            json={"topic": "t_16_recursion_fundamentals", "mode_choice": "diagnostic"}
        )
        assert res_mode_id.status_code == 200
        assert res_mode_id.json()["mode_choice"] == "diagnostic"

        # 3. Set mode with ambiguous name + category_id
        res_mode_cat = await client.post(
            "/api/topics/mode",
            json={
                "topic": "Recursion Fundamentals",
                "category_id": "cat_16_recursion",
                "mode_choice": "beginner"
            }
        )
        assert res_mode_cat.status_code == 200
        assert res_mode_cat.json()["mode_choice"] == "beginner"

        # 4. Set mode with ambiguous name alone -> auto-disambiguates from selected user_topic
        res_mode_auto = await client.post(
            "/api/topics/mode",
            json={"topic": "Recursion Fundamentals", "mode_choice": "diagnostic"}
        )
        assert res_mode_auto.status_code == 200
        assert res_mode_auto.json()["mode_choice"] == "diagnostic"
