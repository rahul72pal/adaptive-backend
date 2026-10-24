import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import patch
import uuid

from main import app
from app.catalog.models import TopicMetadata, CategoryMetadata
from app.services.topic_catalog_service import get_catalog_service
from app.core.dependencies import get_current_user
from app.db.models import User

mock_user = User(
    id=uuid.uuid4(),
    email="catalog_test@example.com",
    name="Catalog Tester",
    hashed_password="mock_hash"
)

@pytest.fixture
def mock_auth():
    app.dependency_overrides[get_current_user] = lambda: mock_user
    yield
    app.dependency_overrides.pop(get_current_user, None)


# -----------------------------------------------------------------------------
# Unit Tests: Catalog Service, Preservation, Uniqueness & Metadata Validation
# -----------------------------------------------------------------------------

def test_catalog_preservation_and_integrity():
    service = get_catalog_service()
    categories = service.get_all_categories()
    topics = service.get_all_topics()

    # Rule 1 & Rule 8: All 60 categories and full topic catalog preserved
    assert len(categories) == 60, f"Expected 60 categories, got {len(categories)}"
    assert len(topics) >= 1700, f"Expected >= 1700 topics, got {len(topics)}"

    # Rule 2: Topic IDs are unique and stable
    topic_ids = [t.topic_id for t in topics]
    assert len(topic_ids) == len(set(topic_ids)), "Duplicate topic_id found in catalog!"

    # Every topic must have valid stable format: t_<cat>_<slug>
    for t in topics:
        assert t.topic_id.startswith("t_"), f"Invalid topic_id format: {t.topic_id}"
        assert t.name, f"Topic {t.topic_id} has empty name"
        assert t.category_id, f"Topic {t.topic_id} has empty category_id"
        assert t.category_id in service._categories_by_id, f"Invalid category_id {t.category_id}"
        assert t.kind in ["data_structure", "algorithm", "technique", "concrete_problem", "math_theory", "systems_practice", "proof_skill", "general"]
        assert t.assessment_mode in ["code", "conceptual_only", "mixed"]
        assert len(t.supported_levels) > 0
        assert t.generation_config.target_node_count >= 3
        assert t.metadata_version >= 1


def test_category_ordering_and_reference():
    service = get_catalog_service()
    categories = service.get_all_categories()

    # Verify category numbers 1 through 60 are sequential
    cat_nums = [c.category_number for c in categories]
    assert cat_nums == list(range(1, 61))

    # Test category retrieval by ID
    cat_05 = service.get_category_by_id("cat_05")
    assert cat_05 is not None
    assert cat_05.name == "Arrays"
    assert cat_05.full_name == "05. Arrays"
    assert cat_05.topic_count > 0


def test_metadata_defaults_and_filtering():
    service = get_catalog_service()

    # Filter by category
    array_topics, total_arr = service.get_topics(category_id="cat_05", page_size=200)
    assert total_arr > 30
    assert all(t.category_id == "cat_05" for t in array_topics)

    # Filter by kind
    ds_topics, total_ds = service.get_topics(kind="data_structure", page_size=50)
    assert total_ds > 0
    assert all(t.kind == "data_structure" for t in ds_topics)

    # Search keyword
    bs_topics, total_bs = service.get_topics(search="Binary Search", page_size=50)
    assert total_bs > 0
    assert any("binary search" in t.name.lower() for t in bs_topics)


def test_topic_resolver_exact_and_legacy_compatibility():
    service = get_catalog_service()

    # Exact topic_id resolution
    res_id = service.resolve_topic("t_08_binary_search")
    assert res_id.is_resolved is True
    assert res_id.topic.topic_id == "t_08_binary_search"
    assert res_id.topic.name == "Binary Search"

    # Legacy display name resolution (single match)
    res_name = service.resolve_topic("Two Sum in Sorted Array")
    assert res_name.is_resolved is True
    assert res_name.topic.category_id == "cat_10"

    # Legacy hyphen slug resolution
    res_slug = service.resolve_topic("two-sum-in-sorted-array")
    assert res_slug.is_resolved is True


def test_topic_resolver_ambiguous_and_unknown_safety():
    service = get_catalog_service()

    # "Modular Arithmetic" appears in 3 categories (03, 43, 58)
    res_ambig = service.resolve_topic("Modular Arithmetic")
    assert res_ambig.is_resolved is False
    assert res_ambig.is_ambiguous is True
    assert len(res_ambig.matching_candidates) >= 2
    assert "Ambiguous topic name" in res_ambig.message

    # Disambiguation with preferred category succeeds
    res_disambig = service.resolve_topic("Modular Arithmetic", preferred_category_id="cat_43")
    assert res_disambig.is_resolved is True
    assert res_disambig.topic.category_id == "cat_43"

    # Unknown topic fails safely
    res_unknown = service.resolve_topic("NonExistentTopicXyz123")
    assert res_unknown.is_resolved is False
    assert res_unknown.is_ambiguous is False
    assert "was not found" in res_unknown.message


# -----------------------------------------------------------------------------
# API Endpoints Integration Tests
# -----------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_api_categories_list():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        res = await client.get("/api/topics/categories")
        assert res.status_code == 200
        data = res.json()
        assert data["total_categories"] == 60
        assert len(data["categories"]) == 60
        first_cat = data["categories"][0]
        assert first_cat["category_id"] == "cat_01"
        assert first_cat["name"] == "Programming Fundamentals"


@pytest.mark.asyncio
async def test_api_catalog_query_and_pagination():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        # Default page 1, 20 items
        res = await client.get("/api/topics/catalog?page=1&page_size=20")
        assert res.status_code == 200
        data = res.json()
        assert data["total_topics"] >= 1700
        assert len(data["topics"]) == 20

        # Filter by category
        res_cat = await client.get("/api/topics/catalog?category_id=cat_10")
        assert res_cat.status_code == 200
        cat_data = res_cat.json()
        assert cat_data["total_topics"] > 0
        assert all(t["category_id"] == "cat_10" for t in cat_data["topics"])

        # Search query
        res_search = await client.get("/api/topics/catalog?search=traversal")
        assert res_search.status_code == 200
        search_data = res_search.json()
        assert search_data["total_topics"] > 0

        # Full catalog retrieval with page_size=2000 (used by frontend search picker)
        res_full = await client.get("/api/topics/catalog?page_size=2000")
        assert res_full.status_code == 200
        full_data = res_full.json()
        assert full_data["total_topics"] >= 1700
        assert len(full_data["topics"]) == full_data["total_topics"]


@pytest.mark.asyncio
async def test_api_get_topic_by_id_and_metadata():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        # Get by stable ID
        res = await client.get("/api/topics/catalog/t_08_binary_search")
        assert res.status_code == 200
        topic_data = res.json()
        assert topic_data["topic_id"] == "t_08_binary_search"
        assert topic_data["name"] == "Binary Search"
        assert topic_data["category_id"] == "cat_08"

        # Metadata endpoint
        res_meta = await client.get("/api/topics/catalog/t_08_binary_search/metadata")
        assert res_meta.status_code == 200
        meta = res_meta.json()
        assert meta["topic_id"] == "t_08_binary_search"
        assert meta["target_node_count"] == 8

        # Nonexistent topic -> 404
        res_404 = await client.get("/api/topics/catalog/t_invalid_missing_id")
        assert res_404.status_code == 404


@pytest.mark.asyncio
async def test_api_resolve_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        # Unambiguous
        res = await client.get("/api/topics/resolve?topic=Two Sum in Sorted Array")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "resolved"
        assert data["result"]["is_resolved"] is True
        assert data["result"]["topic"]["category_id"] == "cat_10"

        # Ambiguous
        res_ambig = await client.get("/api/topics/resolve?topic=Modular Arithmetic")
        assert res_ambig.status_code == 200
        ambig_data = res_ambig.json()
        assert ambig_data["status"] == "ambiguous"
        assert ambig_data["result"]["is_ambiguous"] is True
        assert len(ambig_data["result"]["matching_candidates"]) >= 2


@pytest.mark.asyncio
async def test_api_select_topic_backward_compatibility(mock_auth):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        # 1. Select by stable topic_id
        res_id = await client.post("/api/topics/select", json={"topic": "t_08_binary_search"})
        assert res_id.status_code == 200
        data_id = res_id.json()
        assert "topic_slug" in data_id
        assert data_id["canonical_topic"] == "Binary Search"
        assert data_id["topic_id"] == "t_08_binary_search"
        assert data_id["category_id"] == "cat_08"

        # 2. Select by legacy name
        res_name = await client.post("/api/topics/select", json={"topic": "Two Sum in Sorted Array"})
        assert res_name.status_code == 200
        data_name = res_name.json()
        assert data_name["canonical_topic"] == "Two Sum in Sorted Array"
        assert data_name["category_id"] == "cat_10"

        # 3. Select ambiguous topic name -> 400 validation error
        res_ambig = await client.post("/api/topics/select", json={"topic": "Modular Arithmetic"})
        assert res_ambig.status_code == 400
        assert "Ambiguous topic name" in res_ambig.json()["detail"]

        # 4. Select ambiguous topic with disambiguating category_id -> succeeds
        res_disambig = await client.post(
            "/api/topics/select",
            json={"topic": "Modular Arithmetic", "category_id": "cat_43"}
        )
        assert res_disambig.status_code == 200
        assert res_disambig.json()["category_id"] == "cat_43"


@pytest.mark.asyncio
async def test_api_catalog_sync_idempotency(mock_auth):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        res = await client.post("/api/topics/catalog/sync")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "success"
