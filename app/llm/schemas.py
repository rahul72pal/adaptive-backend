from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class DiagnosticQuestionSchema(BaseModel):
    question: str
    target_concept: str

class ScoreEvaluationSchema(BaseModel):
    confidence: float = Field(ge=0.0, le=1.0)
    status: str  # known, partial, unknown
    feedback: Optional[str] = None

class ConceptNodeItem(BaseModel):
    id: Optional[str] = None
    concept_id: Optional[str] = None
    level: str = "beginner"
    concept_type: str = "conceptual"
    status: Optional[str] = "learning"
    action: Optional[str] = "full_teach"
    estimated_minutes: Optional[int] = 15
    depends_on: List[str] = Field(default_factory=list)

    def model_post_init(self, __context: Any) -> None:
        if not self.id and self.concept_id:
            self.id = self.concept_id
        elif not self.concept_id and self.id:
            self.concept_id = self.id

class ConceptGraphSchema(BaseModel):
    topic: str
    target_level: Optional[str] = "expert"
    aliases: List[str] = Field(default_factory=list)
    nodes: Optional[List[ConceptNodeItem]] = None
    concepts: Optional[List[ConceptNodeItem]] = None

    def model_post_init(self, __context: Any) -> None:
        if self.nodes is None and self.concepts is not None:
            self.nodes = self.concepts
        elif self.concepts is None and self.nodes is not None:
            self.concepts = self.nodes

class GeneratedTestCaseItem(BaseModel):
    input: str
    expected_output: str

class GeneratedQuestionItem(BaseModel):
    type: str  # mcq, true_false, short_answer, open_ended, practice
    concept_id: str
    difficulty: int = Field(ge=1, le=5)
    question: str
    options: Optional[List[str]] = None
    correct_option: Optional[str] = None
    rubric: Optional[str] = None
    test_cases: Optional[List[GeneratedTestCaseItem]] = None
    expected_answer: Optional[str] = None
    execution_method: Optional[str] = "llm_rubric"

class QuestionBatchSchema(BaseModel):
    questions: List[GeneratedQuestionItem]

class RubricEvaluationSchema(BaseModel):
    llm_quality_score: float = Field(ge=0.0, le=1.0)
    feedback: str
    conceptual_breakdown: Dict[str, Any] = Field(default_factory=dict)

class LessonExplanationSchema(BaseModel):
    concept_id: str
    strategy: str  # analogy, visual_diagram, interactive_exercise, worked_example
    content: str
