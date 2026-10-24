from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field, ConfigDict

TopicKind = Literal[
    "data_structure",
    "algorithm",
    "technique",
    "concrete_problem",
    "math_theory",
    "systems_practice",
    "proof_skill",
    "general"
]

TopicScope = Literal[
    "category",
    "topic",
    "leaf"
]

AssessmentMode = Literal[
    "code",
    "conceptual_only",
    "mixed"
]

class GenerationConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")
    target_node_count: int = Field(default=8, ge=3, le=20, description="Target concept nodes in generated graph")
    max_diagnostic_questions: int = Field(default=15, ge=5, le=30, description="Target probe count for diagnostics")
    allow_practice_nodes: bool = Field(default=True, description="Whether practice nodes should be generated")

class TopicMetadata(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    topic_id: str = Field(..., description="Stable, unique canonical topic identifier (e.g. t_08_binary_search)")
    name: str = Field(..., description="Display name of the topic")
    description: str = Field(default="", description="Educational overview and scope of the topic")
    category_id: str = Field(..., description="Stable identifier of parent category (e.g. cat_08)")
    category_name: str = Field(..., description="Canonical category display name (e.g. 08. Searching Algorithms)")
    
    kind: TopicKind = Field(default="technique", description="Taxonomic kind of topic")
    scope: TopicScope = Field(default="topic", description="Granularity scope")
    
    supported_levels: List[str] = Field(
        default_factory=lambda: ["beginner", "intermediate", "advanced", "expert"],
        description="Supported pedagogical proficiency levels"
    )
    default_target_level: str = Field(default="expert", description="Default learning goal target level")
    
    assessment_mode: AssessmentMode = Field(default="mixed", description="Primary assessment format")
    applicable_languages: List[str] = Field(
        default_factory=lambda: ["python", "cpp", "js"],
        description="Supported programming languages for code challenges"
    )
    language_restrictions: Optional[List[str]] = Field(
        default=None,
        description="Optional list of disallowed languages if any"
    )
    
    pattern_tags: List[str] = Field(
        default_factory=list,
        description="DSA algorithmic pattern tags (e.g. binary_search, two_pointer)"
    )
    prerequisite_topic_ids: List[str] = Field(
        default_factory=list,
        description="Stable topic IDs that should precede this topic"
    )
    
    canonical_topic_id: Optional[str] = Field(
        default=None,
        description="Pointer to canonical topic if this entry is an alias"
    )
    aliases: List[str] = Field(
        default_factory=list,
        description="Legacy search aliases, alternative spellings, and historic slugs"
    )
    
    generation_config: GenerationConfig = Field(
        default_factory=GenerationConfig,
        description="LLM generation tuning parameters"
    )
    metadata_version: int = Field(default=1, description="Version of this metadata schema")
    order_in_category: int = Field(default=1, description="Display order within parent category")


class CategoryMetadata(BaseModel):
    model_config = ConfigDict(extra="ignore")

    category_id: str = Field(..., description="Stable category identifier (e.g. cat_05)")
    category_number: int = Field(..., description="1-indexed numeric order of category")
    name: str = Field(..., description="Clean category name (e.g. Arrays)")
    full_name: str = Field(..., description="Original prefixed category name (e.g. 05. Arrays)")
    topic_count: int = Field(default=0, description="Total active topics in this category")
    kind_default: TopicKind = Field(default="technique", description="Default kind for topics in this category")
    assessment_mode_default: AssessmentMode = Field(default="mixed", description="Default assessment mode")
    pattern_tag_default: str = Field(default="general", description="Default pattern tag")
    description: str = Field(default="", description="Summary of topics encompassed by this category")


class CategoryCatalogResponse(BaseModel):
    total_categories: int
    categories: List[CategoryMetadata]


class TopicCatalogResponse(BaseModel):
    total_topics: int
    page: int
    page_size: int
    topics: List[TopicMetadata]


class TopicResolutionResult(BaseModel):
    is_resolved: bool
    is_ambiguous: bool = False
    topic: Optional[TopicMetadata] = None
    matching_candidates: List[TopicMetadata] = Field(default_factory=list)
    message: str = ""


class TopicResolutionResponse(BaseModel):
    status: str
    result: TopicResolutionResult
