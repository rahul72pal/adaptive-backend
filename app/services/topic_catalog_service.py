import os
import json
import logging
import re
from typing import List, Optional, Dict, Tuple, Any
from pathlib import Path

from app.catalog.models import (
    TopicMetadata,
    CategoryMetadata,
    TopicResolutionResult,
    GenerationConfig,
)

logger = logging.getLogger(__name__)

CATALOG_JSON_PATH = Path(__file__).resolve().parent.parent / "catalog" / "data" / "dsa_catalog.json"

class TopicCatalogService:
    """
    Singleton in-memory indexing service for the canonical DSA Topic Catalog.
    Provides O(1) lookups, filtering, pagination, metadata extraction,
    and legacy identifier compatibility resolution.
    """
    _instance: Optional["TopicCatalogService"] = None

    def __init__(self, catalog_path: Optional[Path] = None):
        self.catalog_path = catalog_path or CATALOG_JSON_PATH
        self._categories_by_id: Dict[str, CategoryMetadata] = {}
        self._categories_list: List[CategoryMetadata] = []
        self._topics_by_id: Dict[str, TopicMetadata] = {}
        self._topics_list: List[TopicMetadata] = []
        self._topics_by_category: Dict[str, List[TopicMetadata]] = {}
        self._name_index: Dict[str, List[TopicMetadata]] = {}
        self._alias_index: Dict[str, List[TopicMetadata]] = {}
        self._is_loaded = False
        self._load_catalog()

    @classmethod
    def get_instance(cls) -> "TopicCatalogService":
        if cls._instance is None:
            cls._instance = TopicCatalogService()
        return cls._instance

    def _load_catalog(self) -> None:
        if not self.catalog_path.exists():
            logger.error(f"[TopicCatalogService] Catalog file not found at {self.catalog_path}")
            return

        try:
            with open(self.catalog_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Load Categories
            for cat_dict in data.get("categories", []):
                cat = CategoryMetadata(**cat_dict)
                self._categories_by_id[cat.category_id] = cat
                self._categories_list.append(cat)
                self._topics_by_category[cat.category_id] = []

            # Sort categories by category_number
            self._categories_list.sort(key=lambda c: c.category_number)

            # Load Topics
            for top_dict in data.get("topics", []):
                # Ensure generation_config exists
                if "generation_config" not in top_dict:
                    top_dict["generation_config"] = {
                        "target_node_count": top_dict.get("target_node_count", 8)
                    }
                topic = TopicMetadata(**top_dict)
                self._topics_by_id[topic.topic_id] = topic
                self._topics_list.append(topic)

                if topic.category_id in self._topics_by_category:
                    self._topics_by_category[topic.category_id].append(topic)

                # Index by normalized name
                norm_name = topic.name.strip().lower()
                if norm_name not in self._name_index:
                    self._name_index[norm_name] = []
                self._name_index[norm_name].append(topic)

                # Index by all aliases
                for alias in topic.aliases:
                    clean_alias = alias.strip().lower()
                    if clean_alias not in self._alias_index:
                        self._alias_index[clean_alias] = []
                    if topic not in self._alias_index[clean_alias]:
                        self._alias_index[clean_alias].append(topic)

            self._is_loaded = True
            logger.info(
                f"[TopicCatalogService] Successfully loaded {len(self._categories_list)} categories "
                f"and {len(self._topics_list)} canonical topics from {self.catalog_path.name}."
            )
        except Exception as e:
            logger.error(f"[TopicCatalogService] Failed loading catalog from {self.catalog_path}: {e}", exc_info=True)
            raise

    # -------------------------------------------------------------------------
    # Category Lookups
    # -------------------------------------------------------------------------
    def get_all_categories(self) -> List[CategoryMetadata]:
        return list(self._categories_list)

    def get_category_by_id(self, category_id: str) -> Optional[CategoryMetadata]:
        return self._categories_by_id.get(category_id)

    # -------------------------------------------------------------------------
    # Topic Lookups & Filtered Queries
    # -------------------------------------------------------------------------
    def get_all_topics(self) -> List[TopicMetadata]:
        return list(self._topics_list)

    def get_topic_by_id(self, topic_id: str) -> Optional[TopicMetadata]:
        return self._topics_by_id.get(topic_id)

    def get_topics(
        self,
        category_id: Optional[str] = None,
        kind: Optional[str] = None,
        assessment_mode: Optional[str] = None,
        level: Optional[str] = None,
        search: Optional[str] = None,
        page: int = 1,
        page_size: int = 50,
    ) -> Tuple[List[TopicMetadata], int]:
        """
        Queries catalog topics with multi-axis filtering, keyword search,
        and deterministic pagination.
        """
        results = self._topics_list

        # Filter by Category (accepts category_id like "cat_05" or full_name like "05. Arrays")
        if category_id:
            c_clean = category_id.strip().lower()
            matching_cat_id = None
            if c_clean in self._categories_by_id:
                matching_cat_id = c_clean
            else:
                for cat in self._categories_list:
                    if cat.full_name.lower() == c_clean or cat.name.lower() == c_clean:
                        matching_cat_id = cat.category_id
                        break
            if matching_cat_id and matching_cat_id in self._topics_by_category:
                results = self._topics_by_category[matching_cat_id]
            else:
                results = []

        # Filter by Kind
        if kind and results:
            k_clean = kind.strip().lower()
            results = [t for t in results if t.kind.lower() == k_clean]

        # Filter by Assessment Mode
        if assessment_mode and results:
            am_clean = assessment_mode.strip().lower()
            results = [t for t in results if t.assessment_mode.lower() == am_clean]

        # Filter by Supported Level
        if level and results:
            lvl_clean = level.strip().lower()
            results = [t for t in results if lvl_clean in [l.lower() for l in t.supported_levels]]

        # Free-text Keyword Search
        if search and search.strip() and results:
            q_clean = search.strip().lower()
            words = q_clean.split()
            filtered = []
            for t in results:
                haystack = f"{t.name} {t.category_name} {' '.join(t.pattern_tags)} {' '.join(t.aliases)}".lower()
                if all(w in haystack for w in words):
                    filtered.append(t)
            results = filtered

        total = len(results)
        start_idx = max(0, (page - 1) * page_size)
        end_idx = start_idx + page_size
        return results[start_idx:end_idx], total

    # -------------------------------------------------------------------------
    # Backward Compatibility & Disambiguation Resolver
    # -------------------------------------------------------------------------
    def resolve_topic(
        self,
        raw_topic_input: str,
        preferred_category_id: Optional[str] = None
    ) -> TopicResolutionResult:
        """
        Resolves any incoming topic string (stable ID, slug, display name, or alias)
        to its canonical TopicMetadata record.
        
        Rules:
        1. Exact topic_id match -> resolves immediately.
        2. Exact alias match with 1 candidate -> resolves immediately.
        3. Exact name match with 1 candidate -> resolves immediately.
        4. Multiple matching candidates across categories ->
           - If preferred_category_id matches one, disambiguates to it.
           - Else returns is_ambiguous=True with candidates list and actionable error.
        5. Safe normalized fallback matching.
        """
        input_clean = raw_topic_input.strip()
        if not input_clean:
            return TopicResolutionResult(
                is_resolved=False,
                is_ambiguous=False,
                message="Topic identifier cannot be empty."
            )

        # 1. Exact topic_id match
        if input_clean in self._topics_by_id:
            return TopicResolutionResult(
                is_resolved=True,
                topic=self._topics_by_id[input_clean],
                message="Resolved by exact stable topic_id."
            )

        lower_input = input_clean.lower()
        slug_input = re.sub(r'[^a-z0-9]+', '_', lower_input).strip('_')
        hyphen_slug = lower_input.replace('_', '-')

        # 2. Check alias and name index
        candidates: List[TopicMetadata] = []
        for key in (lower_input, slug_input, hyphen_slug):
            if key in self._alias_index:
                for t in self._alias_index[key]:
                    if t not in candidates:
                        candidates.append(t)
            if key in self._name_index:
                for t in self._name_index[key]:
                    if t not in candidates:
                        candidates.append(t)

        # 3. Disambiguate if preferred category provided
        if len(candidates) > 1 and preferred_category_id:
            p_clean = preferred_category_id.strip().lower()
            cat_matches = [
                t for t in candidates
                if t.category_id.lower() == p_clean
                or t.category_name.lower() == p_clean
                or p_clean in t.category_id.lower()
                or t.category_id.lower() in p_clean
                or p_clean in t.category_name.lower()
            ]
            if len(cat_matches) == 1:
                return TopicResolutionResult(
                    is_resolved=True,
                    topic=cat_matches[0],
                    message=f"Disambiguated '{input_clean}' via preferred category '{preferred_category_id}'."
                )

        # 4. Single candidate match
        if len(candidates) == 1:
            return TopicResolutionResult(
                is_resolved=True,
                topic=candidates[0],
                message=f"Resolved topic name '{input_clean}' to canonical topic '{candidates[0].topic_id}'."
            )

        # 5. Multiple candidates -> Ambiguity error
        if len(candidates) > 1:
            cat_names = [f"'{t.category_name}' (ID: {t.topic_id})" for t in candidates]
            msg = (
                f"Ambiguous topic name '{input_clean}' matched {len(candidates)} distinct catalog topics: "
                f"{', '.join(cat_names)}. Please provide the exact 'topic_id' or 'category_id' to disambiguate."
            )
            return TopicResolutionResult(
                is_resolved=False,
                is_ambiguous=True,
                matching_candidates=candidates,
                message=msg
            )

        # 6. Normalized word boundary scan fallback
        fuzzy_matches: List[TopicMetadata] = []
        for t in self._topics_list:
            t_norm = t.name.lower()
            if lower_input == t_norm or slug_input == re.sub(r'[^a-z0-9]+', '_', t_norm):
                if t not in fuzzy_matches:
                    fuzzy_matches.append(t)

        if len(fuzzy_matches) == 1:
            return TopicResolutionResult(
                is_resolved=True,
                topic=fuzzy_matches[0],
                message=f"Resolved via normalized match to '{fuzzy_matches[0].topic_id}'."
            )
        elif len(fuzzy_matches) > 1:
            return TopicResolutionResult(
                is_resolved=False,
                is_ambiguous=True,
                matching_candidates=fuzzy_matches,
                message=f"Topic input '{input_clean}' matches multiple topics. Please select by exact topic_id."
            )

        return TopicResolutionResult(
            is_resolved=False,
            is_ambiguous=False,
            message=f"Topic '{input_clean}' was not found in the canonical DSA catalog."
        )

    # -------------------------------------------------------------------------
    # Topic Metadata for Curriculum & Diagnostic LLM Prompts
    # -------------------------------------------------------------------------
    def get_topic_metadata_for_generation(self, topic_identifier: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves complete metadata needed for diagnostic probing, curriculum graph
        generation, and adaptive instruction.
        """
        resolution = self.resolve_topic(topic_identifier)
        if not resolution.is_resolved or not resolution.topic:
            return None

        t = resolution.topic
        return {
            "topic_id": t.topic_id,
            "canonical_topic": t.name,
            "category_id": t.category_id,
            "category_name": t.category_name,
            "kind": t.kind,
            "scope": t.scope,
            "assessment_mode": t.assessment_mode,
            "pattern_tags": t.pattern_tags,
            "supported_levels": t.supported_levels,
            "default_target_level": t.default_target_level,
            "applicable_languages": t.applicable_languages,
            "prerequisites": t.prerequisite_topic_ids,
            "target_node_count": t.generation_config.target_node_count,
            "max_diagnostic_questions": t.generation_config.max_diagnostic_questions,
            "description": t.description,
            "metadata_version": t.metadata_version,
        }

    async def sync_to_database(self, db: Optional[Any] = None, mongo_db: Optional[Any] = None) -> Dict[str, int]:
        """
        Idempotently backfills catalog records into PostgreSQL and MongoDB without
        touching any user progress, mastery, sessions, or attempts.
        """
        synced_pg = 0
        synced_mongo = 0

        if db is not None:
            try:
                from sqlalchemy import select
                from app.db.models import TopicCatalog
                res = await db.execute(select(TopicCatalog.id).limit(1))
                existing_one = res.scalar_one_or_none()
                if not existing_one:
                    for t in self._topics_list:
                        catalog_row = TopicCatalog(
                            id=t.topic_id,
                            name=t.name,
                            description=t.description,
                            category_id=t.category_id,
                            category_name=t.category_name,
                            kind=t.kind,
                            scope=t.scope,
                            supported_levels=t.supported_levels,
                            default_target_level=t.default_target_level,
                            assessment_mode=t.assessment_mode,
                            applicable_languages=t.applicable_languages,
                            pattern_tags=t.pattern_tags,
                            prerequisite_topic_ids=t.prerequisite_topic_ids,
                            aliases=t.aliases,
                            target_node_count=t.generation_config.target_node_count,
                            metadata_version=t.metadata_version,
                            metadata_json=t.model_dump()
                        )
                        db.add(catalog_row)
                        synced_pg += 1
                    await db.commit()
                    logger.info(f"[TopicCatalogService] Seeded {synced_pg} topic records to PostgreSQL.")
            except Exception as e:
                logger.warning(f"[TopicCatalogService] PostgreSQL sync skipped/failed: {e}")

        if mongo_db is not None:
            try:
                coll = mongo_db.get_collection("topic_catalog")
                existing_count = await coll.count_documents({})
                if existing_count == 0:
                    docs = []
                    for t in self._topics_list:
                        doc = t.model_dump()
                        doc["_id"] = t.topic_id
                        docs.append(doc)
                    if docs:
                        await coll.insert_many(docs)
                        synced_mongo = len(docs)
                        logger.info(f"[TopicCatalogService] Seeded {synced_mongo} topic records to MongoDB.")
            except Exception as e:
                logger.warning(f"[TopicCatalogService] MongoDB sync skipped/failed: {e}")

        return {"postgres_synced": synced_pg, "mongo_synced": synced_mongo}


def get_catalog_service() -> TopicCatalogService:
    """Dependency helper for FastAPI endpoints."""
    return TopicCatalogService.get_instance()
