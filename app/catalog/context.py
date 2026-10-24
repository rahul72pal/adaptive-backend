"""
Topic Generation Context for metadata-driven LLM pipeline.
Encapsulates all topic-specific pedagogical parameters, constraints,
and learner states required across diagnostics, curriculum graph generation,
lesson delivery, practice question generation, and assessment.
"""
from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field, ConfigDict, model_validator
import logging

from app.catalog.models import TopicMetadata, TopicKind, TopicScope, AssessmentMode

logger = logging.getLogger(__name__)


class TopicGenerationContext(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    # Topic Identification & Taxonomy
    topic_id: str = Field(..., description="Stable canonical topic ID (e.g. t_08_binary_search)")
    canonical_topic: str = Field(..., description="Canonical display name of topic")
    category_id: str = Field(..., description="Parent category ID (e.g. cat_08)")
    category_name: str = Field(..., description="Parent category name")
    description: str = Field(default="", description="Educational scope of the topic")
    kind: TopicKind = Field(default="technique", description="Taxonomic kind of topic")
    scope: TopicScope = Field(default="topic", description="Granularity scope: category, topic, leaf")
    
    # Target Learner Context
    target_level: str = Field(default="intermediate", description="Target learner level")
    supported_levels: List[str] = Field(
        default_factory=lambda: ["beginner", "intermediate", "advanced", "expert"]
    )
    diagnostic_profile: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Diagnostic result summary (weaknesses, strengths, verified level)"
    )
    
    # Assessment & Code Context
    assessment_mode: AssessmentMode = Field(default="mixed", description="Assessment mode")
    selected_language: Optional[str] = Field(
        default=None,
        description="Learner's selected coding language (e.g. python, cpp, js)"
    )
    applicable_languages: List[str] = Field(
        default_factory=lambda: ["python", "cpp", "js"]
    )
    language_restrictions: Optional[List[str]] = Field(default=None)

    # Pedagogical Connections
    pattern_tags: List[str] = Field(default_factory=list)
    prerequisites: List[str] = Field(default_factory=list)

    # Learner History & Mastery
    existing_mastery: Optional[Dict[str, float]] = Field(
        default=None,
        description="Mapping of concept_id to existing mastery score (0.0 to 1.0)"
    )
    prior_learning_state: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Prior completed concepts or nodes"
    )

    # Generation Tuning Constraints
    target_node_count: int = Field(default=8, ge=3, le=20)
    max_diagnostic_questions: int = Field(default=15, ge=5, le=30)
    allow_practice_nodes: bool = Field(default=True)
    
    # Versioning
    metadata_version: int = Field(default=1)
    prompt_schema_version: str = Field(default="v2")

    @model_validator(mode="after")
    def validate_and_normalize(self) -> "TopicGenerationContext":
        # 1. Level validation
        norm_level = self.target_level.lower().strip()
        if norm_level not in [lvl.lower() for lvl in self.supported_levels]:
            logger.warning(
                f"[TopicGenerationContext] Level '{self.target_level}' not in supported levels "
                f"{self.supported_levels} for topic '{self.topic_id}'. Defaulting to 'intermediate'."
            )
            self.target_level = "intermediate"
        else:
            self.target_level = norm_level

        # 2. Language validation
        if self.selected_language:
            norm_lang = self.selected_language.lower().strip()
            # If language restrictions exist and selected language is restricted
            if self.language_restrictions and norm_lang in [lr.lower() for lr in self.language_restrictions]:
                raise ValueError(
                    f"Selected language '{self.selected_language}' is restricted for topic '{self.canonical_topic}'."
                )
            
            # If assessment mode is conceptual_only, language is not required but harmless
            if self.assessment_mode == "conceptual_only":
                # Keep it or mark as conceptual
                pass
            elif norm_lang not in [al.lower() for al in self.applicable_languages]:
                # Non-applicable language chosen
                logger.warning(
                    f"[TopicGenerationContext] Language '{self.selected_language}' not in applicable "
                    f"languages {self.applicable_languages} for '{self.topic_id}'. Falling back to default."
                )
                self.selected_language = self.applicable_languages[0] if self.applicable_languages else "python"
            else:
                self.selected_language = norm_lang
        else:
            # Default language if code or mixed assessment
            if self.assessment_mode in ("code", "mixed") and self.applicable_languages:
                self.selected_language = self.applicable_languages[0]

        return self

    def to_llm_topic_prompt_block(self) -> str:
        """
        Renders a concise, token-efficient metadata instruction block for LLM prompts.
        Prevents full catalog leakage while enforcing accurate taxonomic scope.
        """
        lines = [
            f"Target Topic: {self.canonical_topic} (ID: {self.topic_id})",
            f"Category: {self.category_name}",
            f"Topic Kind: {self.kind} | Granularity Scope: {self.scope}",
            f"Pedagogical Level: {self.target_level}",
            f"Assessment Mode: {self.assessment_mode}",
        ]
        if self.description:
            lines.append(f"Educational Scope: {self.description}")
        if self.assessment_mode in ("code", "mixed") and self.selected_language:
            lines.append(f"Selected Implementation Language: {self.selected_language}")
        if self.pattern_tags:
            lines.append(f"Pattern Tags: {', '.join(self.pattern_tags)}")
        if self.prerequisites:
            lines.append(f"Prerequisite Topics: {', '.join(self.prerequisites)}")
        if self.existing_mastery:
            mastered = [cid for cid, m in self.existing_mastery.items() if m >= 0.8]
            weak = [cid for cid, m in self.existing_mastery.items() if m < 0.5]
            if mastered:
                lines.append(f"Already Mastered Concepts: {', '.join(mastered)}")
            if weak:
                lines.append(f"Demonstrated Weak Concepts: {', '.join(weak)}")
        if self.diagnostic_profile:
            diag_level = self.diagnostic_profile.get("assessed_level")
            gaps = self.diagnostic_profile.get("knowledge_gaps", [])
            if diag_level:
                lines.append(f"Assessed Diagnostic Level: {diag_level}")
            if gaps:
                lines.append(f"Target Knowledge Gaps: {', '.join(gaps[:5])}")

        if self.assessment_mode == "conceptual_only":
            lines.append("Assessment Mode Directive: Theoretical & Conceptual Integrity (No mandatory executable code nodes).")

        return "\n".join(lines)

    def format_prompt_context_block(self) -> str:
        """Alias for to_llm_topic_prompt_block."""
        return self.to_llm_topic_prompt_block()

    @classmethod
    def from_topic_metadata(
        cls,
        topic_metadata: TopicMetadata,
        target_level: str = "intermediate",
        selected_language: Optional[str] = None,
        diagnostic_profile: Optional[Dict[str, Any]] = None,
        existing_mastery: Optional[Dict[str, float]] = None,
        prior_learning_state: Optional[Dict[str, Any]] = None,
        target_node_count: Optional[int] = None,
        max_diagnostic_questions: Optional[int] = None,
    ) -> "TopicGenerationContext":
        """Constructs a TopicGenerationContext directly from a TopicMetadata instance."""
        t = topic_metadata
        node_count = target_node_count if target_node_count is not None else t.generation_config.target_node_count
        diag_q_count = max_diagnostic_questions if max_diagnostic_questions is not None else t.generation_config.max_diagnostic_questions

        return cls(
            topic_id=t.topic_id,
            canonical_topic=t.name,
            category_id=t.category_id,
            category_name=t.category_name,
            description=t.description,
            kind=t.kind,
            scope=t.scope,
            target_level=target_level or t.default_target_level,
            supported_levels=t.supported_levels,
            assessment_mode=t.assessment_mode,
            selected_language=selected_language or (t.applicable_languages[0] if t.applicable_languages else None),
            applicable_languages=t.applicable_languages,
            language_restrictions=t.language_restrictions,
            pattern_tags=t.pattern_tags,
            prerequisites=t.prerequisite_topic_ids,
            existing_mastery=existing_mastery,
            prior_learning_state=prior_learning_state,
            target_node_count=node_count,
            max_diagnostic_questions=diag_q_count,
            allow_practice_nodes=t.generation_config.allow_practice_nodes,
            diagnostic_profile=diagnostic_profile,
            metadata_version=t.metadata_version,
        )

    @classmethod
    def from_topic_identifier(
        cls,
        topic_identifier: str,
        target_level: str = "intermediate",
        selected_language: Optional[str] = None,
        diagnostic_profile: Optional[Dict[str, Any]] = None,
        existing_mastery: Optional[Dict[str, float]] = None,
        prior_learning_state: Optional[Dict[str, Any]] = None,
        target_node_count: Optional[int] = None,
        max_diagnostic_questions: Optional[int] = None,
    ) -> "TopicGenerationContext":
        """
        Resolves topic identifier against the canonical catalog and instantiates
        a validated TopicGenerationContext.
        """
        from app.services.topic_catalog_service import get_catalog_service
        catalog_svc = get_catalog_service()
        resolution = catalog_svc.resolve_topic(topic_identifier)

        if not resolution.is_resolved or not resolution.topic:
            # Fallback for dynamic/unregistered edge cases
            logger.warning(
                f"[TopicGenerationContext] Could not resolve topic '{topic_identifier}' in catalog: {resolution.message}. "
                "Constructing fallback context."
            )
            # Create a safe fallback
            clean_name = topic_identifier.strip().replace("_", " ").title()
            return cls(
                topic_id=f"custom_{topic_identifier.strip().lower()}",
                canonical_topic=clean_name,
                category_id="cat_custom",
                category_name="Data Structures & Algorithms",
                description=f"Curriculum on {clean_name}",
                kind="technique",
                scope="topic",
                target_level=target_level,
                assessment_mode="mixed",
                selected_language=selected_language or "python",
                pattern_tags=["general"],
                prerequisites=[],
                target_node_count=target_node_count or 8,
                max_diagnostic_questions=max_diagnostic_questions or 15,
                allow_practice_nodes=True,
                diagnostic_profile=diagnostic_profile,
                existing_mastery=existing_mastery,
                prior_learning_state=prior_learning_state,
            )

        t = resolution.topic
        node_count = target_node_count if target_node_count is not None else t.generation_config.target_node_count
        diag_q_count = max_diagnostic_questions if max_diagnostic_questions is not None else t.generation_config.max_diagnostic_questions

        return cls(
            topic_id=t.topic_id,
            canonical_topic=t.name,
            category_id=t.category_id,
            category_name=t.category_name,
            description=t.description,
            kind=t.kind,
            scope=t.scope,
            target_level=target_level or t.default_target_level,
            supported_levels=t.supported_levels,
            assessment_mode=t.assessment_mode,
            selected_language=selected_language or (t.applicable_languages[0] if t.applicable_languages else None),
            applicable_languages=t.applicable_languages,
            language_restrictions=t.language_restrictions,
            pattern_tags=t.pattern_tags,
            prerequisites=t.prerequisite_topic_ids,
            existing_mastery=existing_mastery,
            prior_learning_state=prior_learning_state,
            target_node_count=node_count,
            max_diagnostic_questions=diag_q_count,
            allow_practice_nodes=t.generation_config.allow_practice_nodes,
            diagnostic_profile=diagnostic_profile,
            metadata_version=t.metadata_version,
        )
