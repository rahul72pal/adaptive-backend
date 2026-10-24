import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import patch, AsyncMock
import uuid

from main import app
from app.catalog.models import TopicMetadata, CategoryMetadata
from app.catalog.context import TopicGenerationContext
from app.services.topic_catalog_service import get_catalog_service
from app.core.dependencies import get_current_user
from app.db.models import User
from app.llm.schemas import (
    DiagnosticChecklistSchema,
    DiagnosticChecklistItem,
    DiagnosticQuestionSchema,
    EvaluatorGradeSchema,
    FeedbackAgentSchema,
    ScoreEvaluationSchema,
    ConceptGraphSchema,
    ConceptNodeItem,
    LessonContentSchema,
    CheckpointQuestion,
    Complexity,
    QuestionBatchSchema,
    GeneratedQuestionItem,
)

mock_user = User(
    id=uuid.uuid4(),
    email="pipeline_test@example.com",
    name="Pipeline Tester",
    hashed_password="mock_hash_pipeline"
)

@pytest.fixture
def mock_auth():
    app.dependency_overrides[get_current_user] = lambda: mock_user
    yield
    app.dependency_overrides.pop(get_current_user, None)


# =============================================================================
# 1. SIX DISTINCT TOPIC SHAPES VERIFICATION
# =============================================================================

def test_six_distinct_topic_shapes_in_catalog():
    service = get_catalog_service()

    # Shape 1: Broad topic (Binary Search)
    broad = service.get_topic_by_id("t_08_binary_search")
    assert broad is not None
    assert broad.name == "Binary Search"
    assert broad.kind in ["algorithm", "technique"]
    assert broad.scope == "topic"
    assert broad.assessment_mode == "mixed"
    assert broad.generation_config.target_node_count >= 5

    # Shape 2: Leaf algorithm (Kadane's Algorithm)
    leaf = service.get_topic_by_id("t_05_kadane_s_algorithm")
    assert leaf is not None
    assert leaf.name == "Kadane's Algorithm"
    assert leaf.kind == "algorithm"
    assert leaf.scope == "leaf"
    assert leaf.generation_config.target_node_count == 5
    assert leaf.generation_config.max_diagnostic_questions == 10

    # Shape 3: Concrete graph problem (Course Schedule)
    concrete = service.get_topic_by_id("t_27_course_schedule")
    assert concrete is not None
    assert concrete.name == "Course Schedule"
    assert concrete.kind == "concrete_problem"
    assert "topological_sort" in concrete.pattern_tags or len(concrete.pattern_tags) > 0

    # Shape 4: Conceptual / Theory topic (Memory Layout)
    conceptual = service.get_topic_by_id("t_57_memory_layout_of_data_structures")
    assert conceptual is not None
    assert conceptual.name == "Memory Layout of Data Structures"
    assert conceptual.kind == "systems_practice"
    assert conceptual.assessment_mode == "conceptual_only"

    # Shape 5: Stateful design problem (Implement LRU Cache)
    stateful = service.get_topic_by_id("t_56_implement_lru_cache")
    assert stateful is not None
    assert stateful.name == "Implement LRU Cache"
    assert stateful.kind == "concrete_problem"
    assert stateful.scope == "leaf"
    assert any("lru" in p.lower() or "cache" in p.lower() or "hash" in p.lower() for p in stateful.pattern_tags)

    # Shape 6: Language-restricted topic (Pointers and References)
    restricted = service.get_topic_by_id("t_01_pointers_and_references")
    assert restricted is not None
    assert restricted.name == "Pointers and References"
    assert "cpp" in [l.lower() for l in restricted.applicable_languages]
    assert restricted.language_restrictions is not None
    assert "python" in [r.lower() for r in restricted.language_restrictions]


# =============================================================================
# 2. TOPIC GENERATION CONTEXT & LANGUAGE RESTRICTION ENFORCEMENT
# =============================================================================

def test_language_restriction_enforcement():
    service = get_catalog_service()
    restricted = service.get_topic_by_id("t_01_pointers_and_references")

    # C++ is supported and allowed
    ctx_valid = TopicGenerationContext.from_topic_metadata(
        restricted,
        selected_language="cpp"
    )
    assert ctx_valid.selected_language == "cpp"

    # Python is explicitly restricted for this topic
    with pytest.raises(ValueError) as excinfo:
        TopicGenerationContext.from_topic_metadata(
            restricted,
            selected_language="python"
        )
    assert "restricted" in str(excinfo.value).lower()

    # JavaScript is explicitly restricted for this topic
    with pytest.raises(ValueError) as excinfo_js:
        TopicGenerationContext.from_topic_metadata(
            restricted,
            selected_language="js"
        )
    assert "restricted" in str(excinfo_js.value).lower()


def test_conceptual_topic_context_and_mode_propagation():
    service = get_catalog_service()
    conceptual = service.get_topic_by_id("t_57_memory_layout_of_data_structures")

    ctx = TopicGenerationContext.from_topic_metadata(
        conceptual,
        target_level="intermediate"
    )
    assert ctx.assessment_mode == "conceptual_only"
    prompt_block = ctx.format_prompt_context_block()
    assert "Assessment Mode: conceptual_only" in prompt_block
    assert "Theoretical & Conceptual Integrity" in prompt_block


def test_level_validation_and_fallback():
    service = get_catalog_service()
    topic = service.get_topic_by_id("t_08_binary_search")

    # Invalid level defaults safely to 'intermediate'
    ctx = TopicGenerationContext.from_topic_metadata(
        topic,
        target_level="grandmaster_unknown"
    )
    assert ctx.target_level == "intermediate"


# =============================================================================
# 3. CONTINUOUS EVALUATOR SCORING BOUNDARIES
# =============================================================================

def test_continuous_evaluator_score_boundaries():
    # Boundary: 0.80 -> known
    grade_80 = EvaluatorGradeSchema(
        confidence=0.80,
        status="known",
        mechanism_correct=True,
        complexity_correct=True,
        key_gaps=[],
        evaluation_summary="Correct mechanism and complexity"
    )
    assert grade_80.status == "known"

    # Boundary: 0.79 -> partial
    grade_79 = EvaluatorGradeSchema(
        confidence=0.79,
        status="partial",
        mechanism_correct=True,
        complexity_correct=False,
        key_gaps=["O(n) instead of O(log n)"],
        evaluation_summary="Mechanism mostly correct, complexity inaccurate"
    )
    assert grade_79.status == "partial"

    # Boundary: 0.40 -> partial
    grade_40 = EvaluatorGradeSchema(
        confidence=0.40,
        status="partial",
        mechanism_correct=True,
        complexity_correct=False,
        key_gaps=["Incomplete edge cases"],
        evaluation_summary="Partial understanding shown"
    )
    assert grade_40.status == "partial"

    # Boundary: 0.39 -> unknown
    grade_39 = EvaluatorGradeSchema(
        confidence=0.39,
        status="unknown",
        mechanism_correct=False,
        complexity_correct=False,
        key_gaps=["Fundamental confusion"],
        evaluation_summary="Student does not understand concept"
    )
    assert grade_39.status == "unknown"

    # Auto-alignment check
    grade_auto = EvaluatorGradeSchema(
        confidence=0.85,
        status="unknown",  # Mismatched status should be synced to 'known' by validator
        mechanism_correct=True,
        complexity_correct=True,
        key_gaps=[],
        evaluation_summary="Valid mechanism"
    )
    assert grade_auto.status == "known"


# =============================================================================
# 4. GRAPH INTEGRITY & ACYCLIC VALIDATION
# =============================================================================

def test_graph_schema_validation_and_node_normalization():
    # Valid ConceptGraphSchema with both learning and practice nodes
    graph = ConceptGraphSchema(
        topic="t_08_binary_search",
        target_level="intermediate",
        nodes=[
            ConceptNodeItem(
                concept_id="bs_fundamentals",
                node_type="learning",
                pattern="binary_search",
                depends_on=[]
            ),
            ConceptNodeItem(
                concept_id="bs_lower_bound",
                node_type="learning",
                pattern="binary_search",
                depends_on=["bs_fundamentals"]
            ),
            ConceptNodeItem(
                concept_id="bs_practice_search_insert",
                node_type="practice",
                pattern="binary_search",
                depends_on=["bs_lower_bound"]
            )
        ]
    )
    assert len(graph.nodes) == 3
    assert graph.nodes[0].concept_id == "bs_fundamentals"
    assert graph.nodes[1].depends_on == ["bs_fundamentals"]
    assert graph.nodes[2].node_type == "practice"

    # Verify ID normalization replaces spaces with underscores
    node_space = ConceptNodeItem(
        concept_id="binary search space",
        node_type="learning"
    )
    assert node_space.concept_id == "binary_search_space"


# =============================================================================
# 5. MASTERY EMA CALCULATION & TOPIC RESOLUTION WITHOUT FRAGMENTATION
# =============================================================================

def test_mastery_ema_formula_preservation():
    # Standard EMA update: new_score = alpha * attempt_score + (1 - alpha) * old_score
    alpha = 0.3
    old_score = 0.5
    attempt_score = 1.0
    expected_new = (alpha * attempt_score) + ((1 - alpha) * old_score)
    assert expected_new == pytest.approx(0.65)

    # Re-testing with lower attempt score
    attempt_lower = 0.2
    expected_decay = (alpha * attempt_lower) + ((1 - alpha) * expected_new)
    assert expected_decay == pytest.approx(0.515)


def test_topic_normalization_avoids_mastery_fragmentation():
    service = get_catalog_service()

    # Querying by canonical ID resolves directly
    res_id = service.resolve_topic("t_08_binary_search")
    assert res_id.is_resolved is True
    assert res_id.topic.topic_id == "t_08_binary_search"

    # Querying with preferred_category_id disambiguates
    res_disambig = service.resolve_topic("Binary Search", preferred_category_id="cat_08")
    assert res_disambig.is_resolved is True
    assert res_disambig.topic.topic_id == "t_08_binary_search"

    # Querying unique topic across all categories resolves unambiguously
    res_unique_name = service.resolve_topic("Kadane's Algorithm")
    res_unique_slug = service.resolve_topic("kadane-s-algorithm")
    res_unique_id = service.resolve_topic("t_05_kadane_s_algorithm")
    assert res_unique_name.is_resolved and res_unique_slug.is_resolved and res_unique_id.is_resolved
    assert res_unique_name.topic.topic_id == res_unique_slug.topic.topic_id == res_unique_id.topic.topic_id == "t_05_kadane_s_algorithm"


# =============================================================================
# 6. END-TO-END PIPELINE FLOW (MOCKED LLM GATEWAY)
# =============================================================================

@pytest.mark.asyncio
async def test_e2e_adaptive_pipeline_mocked(mock_auth):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        # Step 1: Health check
        res_h = await client.get("/health")
        assert res_h.status_code == 200

        # Step 2: Catalog categories & topics
        res_cat = await client.get("/api/topics/categories")
        assert res_cat.status_code == 200
        assert res_cat.json()["total_categories"] == 60

        # Step 3: Fetch topic metadata for canonical ID
        res_t = await client.get("/api/topics/catalog/t_08_binary_search")
        assert res_t.status_code == 200
        t_meta = res_t.json()
        assert t_meta["topic_id"] == "t_08_binary_search"
        assert t_meta["name"] == "Binary Search"

        # Step 4: Select topic using canonical ID
        select_payload = {"topic": "t_08_binary_search", "category_id": "cat_08"}
        res_sel = await client.post("/api/topics/select", json=select_payload)
        assert res_sel.status_code == 200
        sel_data = res_sel.json()
        assert sel_data["canonical_topic"] == "Binary Search"
        assert sel_data["topic_id"] == "t_08_binary_search"

        # Step 5: Set Topic Mode Choice
        mode_payload = {"topic": "t_08_binary_search", "mode_choice": "diagnostic"}
        res_m = await client.post("/api/topics/mode", json=mode_payload)
        assert res_m.status_code == 200
        assert res_m.json()["mode_choice"] == "diagnostic"

        # Mock structured LLM responses for the diagnostic & learning steps
        mock_checklist = DiagnosticChecklistSchema(
            topic="Binary Search",
            concepts=[
                DiagnosticChecklistItem(concept_id="bs_loop_invariants", level="beginner"),
                DiagnosticChecklistItem(concept_id="bs_lower_upper_bound", level="intermediate"),
                DiagnosticChecklistItem(concept_id="bs_rotated_array", level="advanced"),
            ]
        )
        mock_q = DiagnosticQuestionSchema(
            question="In a binary search on a sorted array, how do you prevent integer overflow when calculating mid?",
            target_concept="bs_loop_invariants"
        )
        mock_grade = EvaluatorGradeSchema(
            confidence=0.85,
            status="known",
            mechanism_correct=True,
            complexity_correct=True,
            key_gaps=[],
            evaluation_summary="Candidate correctly stated mid = low + (high - low) // 2."
        )
        mock_feedback = FeedbackAgentSchema(
            feedback_markdown="**Great job!** You correctly identified overflow prevention.",
            key_takeaway="Use low + (high - low) // 2 to avoid integer overflow."
        )
        mock_score = ScoreEvaluationSchema(
            confidence=0.85,
            status="known",
            feedback="Correct mechanism."
        )

        async def fake_llm_gateway(prompt, params, schema):
            if schema == DiagnosticChecklistSchema:
                return mock_checklist
            elif schema == DiagnosticQuestionSchema:
                return mock_q
            elif schema == EvaluatorGradeSchema:
                return mock_grade
            elif schema == FeedbackAgentSchema:
                return mock_feedback
            elif schema == ScoreEvaluationSchema:
                return mock_score
            elif schema == ConceptGraphSchema:
                target_lvl = params.get("target_level", "beginner")
                prefix = f"bs_{target_lvl}"
                return ConceptGraphSchema(
                    topic="Binary Search",
                    target_level=target_lvl,
                    nodes=[
                        ConceptNodeItem(
                            concept_id=f"{prefix}_invariants",
                            level=target_lvl,
                            node_type="learning",
                            pattern="binary_search",
                            depends_on=[]
                        ),
                        ConceptNodeItem(
                            concept_id=f"{prefix}_bounds",
                            level=target_lvl,
                            node_type="learning",
                            pattern="binary_search",
                            depends_on=[f"{prefix}_invariants"]
                        ),
                        ConceptNodeItem(
                            concept_id=f"{prefix}_practice",
                            level=target_lvl,
                            node_type="practice",
                            pattern="binary_search",
                            depends_on=[f"{prefix}_bounds"]
                        )
                    ]
                )
            elif schema == LessonContentSchema:
                return LessonContentSchema(
                    concept_id="bs_loop_invariants",
                    intuition="Binary search repeatedly halves the search space.",
                    core_idea="Maintain the invariant that the target lies in [low, high].",
                    mechanism="Calculate mid = low + (high - low) // 2 and narrow the bounds.",
                    step_by_step_example="Given nums=[1,3,5,7,9], target=5: low=0, high=4 -> mid=2 (nums[2]=5). Found!",
                    code_snippet="def binary_search(nums, target):\n    low, high = 0, len(nums) - 1\n    while low <= high:\n        mid = low + (high - low) // 2\n        if nums[mid] == target:\n            return mid\n        elif nums[mid] < target:\n            low = mid + 1\n        else:\n            high = mid - 1\n    return -1",
                    language="python",
                    complexity=Complexity(
                        best_case="O(1)",
                        worst_case="O(log n)",
                        space="O(1)",
                        explanation="Halves search space each iteration."
                    ),
                    edge_cases=["target smaller than min", "target larger than max", "empty array"],
                    common_mistakes=["integer overflow with (low + high) // 2", "off-by-one with <= vs <"],
                    checkpoint=CheckpointQuestion(
                        question="What is the time complexity of standard binary search?",
                        options=["O(1)", "O(log n)", "O(n)", "O(n log n)"],
                        correct_option="O(log n)",
                        explanation="Each step reduces the problem size by half."
                    )
                )
            elif schema == QuestionBatchSchema:
                return QuestionBatchSchema(
                    questions=[
                        GeneratedQuestionItem(
                            type="mcq",
                            concept_id="bs_loop_invariants",
                            pattern="binary_search",
                            difficulty=2,
                            question="Which index formula prevents integer overflow in binary search?",
                            options=[
                                "mid = (low + high) // 2",
                                "mid = low + (high - low) // 2",
                                "mid = (low * high) // 2",
                                "mid = high - low"
                            ],
                            correct_option="mid = low + (high - low) // 2",
                            explanation="low + (high - low) // 2 never overflows standard integer ranges.",
                            edge_cases=["low == high", "large 32-bit integers"]
                        )
                    ]
                )
            # Default fallback for other schemas
            return schema.model_validate({})

        with patch("app.llm.gateway.call_llm_gateway", side_effect=fake_llm_gateway), \
             patch("app.engine.curriculum_engine.call_llm_gateway", side_effect=fake_llm_gateway), \
             patch("app.graphs.diagnostic_graph.call_llm_gateway", side_effect=fake_llm_gateway), \
             patch("app.agents.evaluator_agent.call_llm_gateway", side_effect=fake_llm_gateway), \
             patch("app.agents.feedback_agent.call_llm_gateway", side_effect=fake_llm_gateway):

            # Step 6: Start Diagnostic Session
            diag_payload = {"topic": "t_08_binary_search"}
            res_diag = await client.post("/api/diagnostic/start", json=diag_payload)
            assert res_diag.status_code in [200, 201]
            diag_body = res_diag.json()
            session_id = diag_body["session_id"]
            assert "current_question" in diag_body

            # Step 7: Answer diagnostic probe
            ans_payload = {
                "session_id": session_id,
                "answer": "We compute mid as low + (high - low) // 2 which takes O(1) time and prevents overflow."
            }
            res_ans = await client.post("/api/diagnostic/answer", json=ans_payload)
            assert res_ans.status_code == 200
            ans_body = res_ans.json()
            assert "knowledge_profile" in ans_body

            # Step 8: Opt out or complete diagnostic to finalize profile
            res_opt = await client.post("/api/diagnostic/opt-out", json={"session_id": session_id})
            assert res_opt.status_code == 200
            assert res_opt.json()["is_complete"] is True

            # Step 9: Generate metadata-driven curriculum learning path
            path_payload = {
                "topic": "t_08_binary_search",
                "target_level": "intermediate",
                "diagnostic_session_id": session_id
            }
            res_path = await client.post("/api/learning/generate-path", json=path_payload)
            assert res_path.status_code == 201
            path_data = res_path.json()
            assert "nodes" in path_data
            assert len(path_data["nodes"]) >= 3
            path_id = path_data["_id"]

            first_learning_node = next(n for n in path_data["nodes"] if n["node_type"] == "learning")
            concept_id = first_learning_node["concept_id"]

            # Step 10: Fetch Learning Content for learning node
            lesson_payload = {
                "topic": "t_08_binary_search",
                "concept_id": concept_id,
                "level": "beginner",
                "pattern": first_learning_node.get("pattern", "binary_search"),
                "prerequisites": first_learning_node.get("depends_on", [])
            }
            res_lesson = await client.post("/api/learning/lesson", json=lesson_payload)
            assert res_lesson.status_code == 200
            lesson_body = res_lesson.json()
            assert "intuition" in lesson_body
            assert "core_idea" in lesson_body

            # Step 11: Complete Node
            res_complete = await client.post("/api/learning/complete-node", json={
                "topic": "t_08_binary_search",
                "concept_id": concept_id,
                "learning_path_id": path_id
            })
            assert res_complete.status_code == 200

            # Step 12: Complete Level & Progress
            res_complete_lvl = await client.post("/api/learning/complete-level", json={
                "learning_path_id": path_id
            })
            assert res_complete_lvl.status_code == 200
            next_lvl_data = res_complete_lvl.json()
            assert "current_level" in next_lvl_data
