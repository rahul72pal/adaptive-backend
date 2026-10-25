"""
LLM Structured Output Schemas — Pydantic v2
============================================

Every schema inherits ``StructuredOutputSchema`` and is self-describing:
  - ``format_instructions()``  — compact JSON example the LLM should mirror
  - ``normalize(raw, hint)``   — coerce loosely-shaped LLM output into this schema

No external code should branch on ``schema.__name__``.  All format hints and
normalization rules live next to the schema they belong to.

Groups
------
  1. Diagnostic — concept probing & scoring
  2. Curriculum — graph generation & learning content
  3. Questions  — question generation & evaluation
  4. Re-teach   — adaptive lesson regeneration
"""

from __future__ import annotations

import json
from typing import Any, ClassVar, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ===========================================================================
# Base class — every LLM output schema inherits this
# ===========================================================================

class StructuredOutputSchema(BaseModel):
    """Base for all LLM-output schemas.

    Subclasses declare:
      ``_format_example``  — compact JSON the LLM should mirror
      ``_format_extra``    — extra behavioural instruction beyond the example

    And override ``normalize()`` when the LLM's raw dict needs coercion
    (missing keys, alternate field names, etc.).
    """

    _format_example: ClassVar[str] = ""
    _format_extra: ClassVar[str] = ""

    @classmethod
    def format_instructions(cls) -> str:
        parts = [
            "Output ONLY raw valid JSON matching this exact shape "
            "(no markdown, no backticks, no prose):",
        ]
        if cls._format_example:
            parts.append(cls._format_example)
        if cls._format_extra:
            parts.append(cls._format_extra)
        return "\n".join(parts).strip()

    @classmethod
    def normalize(cls, raw: Any, hint: str = "") -> "StructuredOutputSchema":
        """Coerce loosely-shaped LLM output into this schema.

        Override per concrete class to handle alternate field names,
        missing defaults, etc.  Base implementation does a straight
        ``model_validate``.
        """
        return cls.model_validate(raw)


# ===========================================================================
# 1. Diagnostic
# ===========================================================================

class DiagnosticQuestionSchema(StructuredOutputSchema):
    """Single probing question for a diagnostic session."""
    question: str = Field(..., min_length=5, description="The diagnostic question text")
    target_concept: str = Field(..., min_length=1)

    _format_example: ClassVar[str] = (
        '{"question": "Walk through how a two-pointer approach finds a pair '
        'summing to 9 in [5,3,8,1]. What is the time complexity?", '
        '"target_concept": "array_two_pointer"}'
    )

    @classmethod
    def normalize(cls, raw: Any, hint: str = "") -> "DiagnosticQuestionSchema":
        if isinstance(raw, dict):
            raw.setdefault("target_concept", hint or "concept")
        return cls.model_validate(raw)


class DiagnosticChecklistItem(BaseModel):
    """One concept to probe in a diagnostic session."""
    concept_id: str = Field(..., min_length=1, description="snake_case concept identifier")
    level: Literal["beginner", "intermediate", "advanced", "expert"] = "beginner"

    @field_validator("concept_id")
    @classmethod
    def _normalise_id(cls, v: str) -> str:
        return v.strip().lower().replace(" ", "_")


class DiagnosticChecklistSchema(StructuredOutputSchema):
    """Flat list of ~15 concept_ids to probe during a diagnostic session."""
    topic: str
    concepts: List[DiagnosticChecklistItem] = Field(..., min_length=1)

    _format_example: ClassVar[str] = (
        '{"topic": "string", "concepts": [{"concept_id": "snake_case_string", '
        '"level": "beginner|intermediate|advanced|expert"}]}'
    )

    @classmethod
    def normalize(cls, raw: Any, hint: str = "") -> "DiagnosticChecklistSchema":
        if isinstance(raw, list):
            raw = {"topic": hint or "DSA", "concepts": raw}
        if isinstance(raw, dict):
            raw.setdefault("topic", hint or "DSA")
        return cls.model_validate(raw)


class ScoreEvaluationSchema(StructuredOutputSchema):
    """Result of scoring a student's answer against a concept."""
    confidence: float = Field(..., ge=0.0, le=1.0, description="0.0 = unknown, 1.0 = fully known")
    status: Literal["known", "partial", "unknown", "unassessed"] = "unassessed"
    feedback: Optional[str] = Field(
        default=None,
        description="Concise, constructive feedback naming the gap (mechanism or complexity)"
    )

    _format_example: ClassVar[str] = (
        '{"confidence": <float 0.0 to 1.0>, "status": "known"|"partial"|"unknown", '
        '"feedback": "<concise constructive feedback on mechanism and complexity>"}'
    )

    @model_validator(mode="after")
    def _sync_status_from_confidence(self) -> "ScoreEvaluationSchema":
        """Auto-derive status from confidence if status is still default."""
        if self.status == "unassessed":
            if self.confidence >= 0.8:
                self.status = "known"
            elif self.confidence >= 0.4:
                self.status = "partial"
            else:
                self.status = "unknown"
        return self


class EvaluatorGradeSchema(StructuredOutputSchema):
    """Output of the specialized AnswerEvaluatorAgent."""
    confidence: float = Field(..., ge=0.0, le=1.0, description="0.0 to 1.0 confidence score")
    status: Literal["known", "partial", "unknown"] = Field(..., description="'known' (>=0.8), 'partial' (0.4-0.79), 'unknown' (<0.4)")
    mechanism_correct: bool = Field(..., description="Whether the core algorithm or data structure mechanism is correct")
    complexity_correct: bool = Field(..., description="Whether time and space complexity claims are correct")
    key_gaps: List[str] = Field(default_factory=list, description="Specific gaps or misconceptions identified")
    evaluation_summary: str = Field(..., description="Objective summary of the evaluation")

    _format_example: ClassVar[str] = (
        '{"confidence": <float between 0.0 and 1.0>, "status": "known"|"partial"|"unknown", '
        '"mechanism_correct": <true or false>, "complexity_correct": <true or false>, '
        '"key_gaps": ["<gap description if any>"], "evaluation_summary": "<brief summary of evaluation>"}'
    )

    @model_validator(mode="after")
    def _validate_status_alignment(self) -> "EvaluatorGradeSchema":
        if self.confidence >= 0.8:
            self.status = "known"
        elif self.confidence >= 0.4:
            self.status = "partial"
        else:
            self.status = "unknown"
        return self


class FeedbackAgentSchema(StructuredOutputSchema):
    """Output of the specialized FeedbackAgent."""
    feedback_markdown: str = Field(..., description="Constructive, pedagogical feedback formatted in clean GitHub Flavored Markdown")
    key_takeaway: str = Field(..., description="One concise memorable sentence summarizing the core takeaway")

    _format_example: ClassVar[str] = (
        '{"feedback_markdown": "**Assessment:** ...\\n\\n- **Core Mechanism:** ...\\n- **Complexity:** ...\\n- **Next Steps:** ...", '
        '"key_takeaway": "<key takeaway sentence>"}'
    )




# ===========================================================================
# 2. Curriculum
# ===========================================================================

ConceptType = Literal["code", "math", "conceptual", "mcq", "short_answer"]
NodeType = Literal["learning", "practice"]
NodeStatus = Literal["learning", "practice", "completed", "skipped"]
NodeAction = Literal["full_teach", "reinforce", "skip", "assess_only"]

DSA_PATTERNS = {
    "two_pointer", "sliding_window", "fast_slow_pointer", "dfs", "bfs",
    "backtracking", "dp_1d", "dp_2d", "greedy", "binary_search",
    "prefix_sum", "divide_conquer", "heap", "union_find",
    "bit_manipulation", "topological_sort", "trie", "none",
}


class ConceptNodeItem(BaseModel):
    """A single node in the concept dependency graph."""
    # Accept either 'id' or 'concept_id' — normalised to concept_id
    id: Optional[str] = Field(default=None, exclude=True)
    concept_id: Optional[str] = Field(default=None, min_length=1)

    level: Literal["beginner", "intermediate", "advanced", "expert"] = "beginner"
    node_type: NodeType = "learning"
    concept_type: ConceptType = "conceptual"
    pattern: Optional[str] = None
    status: NodeStatus = "learning"
    action: NodeAction = "full_teach"
    estimated_minutes: int = Field(default=15, ge=1, le=240)
    target_question_count: int = Field(default=0, ge=0, le=50)
    depends_on: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _normalise_ids(self) -> "ConceptNodeItem":
        if not self.concept_id and self.id:
            self.concept_id = self.id
        elif not self.id and self.concept_id:
            self.id = self.concept_id
        if not self.concept_id or not str(self.concept_id).strip():
            raise ValueError("ConceptNodeItem must have a valid non-empty concept_id or id")
        # Normalise
        self.concept_id = self.concept_id.strip().lower().replace(" ", "_")
        self.id = self.concept_id
        return self

    @field_validator("pattern")
    @classmethod
    def _normalise_pattern(cls, v: Optional[str]) -> Optional[str]:
        if v is None or v.strip().lower() in ("", "none", "null"):
            return "none"
        return v.strip().lower()


class ConceptGraphSchema(StructuredOutputSchema):
    """Full concept graph returned by the curriculum engine."""
    topic: str
    target_level: Optional[str] = "expert"
    aliases: List[str] = Field(default_factory=list)
    # Both 'nodes' and 'concepts' are accepted; they are unified post-init
    nodes: Optional[List[ConceptNodeItem]] = None
    concepts: Optional[List[ConceptNodeItem]] = None

    _format_example: ClassVar[str] = (
        '{"topic": "string", "target_level": "string", "aliases": ["string"], '
        '"nodes": [{"concept_id": "string", "level": "string", '
        '"node_type": "learning|practice", '
        '"concept_type": "code|math|conceptual|other", '
        '"pattern": "string|null", "status": "learning", '
        '"action": "full_teach", "estimated_minutes": 15, '
        '"target_question_count": 0, "depends_on": ["string"]}]}'
    )

    @model_validator(mode="after")
    def _unify_nodes_concepts(self) -> "ConceptGraphSchema":
        if self.nodes is None and self.concepts is not None:
            self.nodes = self.concepts
        elif self.concepts is None and self.nodes is not None:
            self.concepts = self.nodes
        if not self.nodes and not self.concepts:
            raise ValueError(
                "ConceptGraphSchema requires at least one node in 'nodes' or 'concepts'."
            )
        return self

    @classmethod
    def normalize(cls, raw: Any, hint: str = "") -> "ConceptGraphSchema":
        if isinstance(raw, list):
            raw = {"topic": hint or "DSA", "nodes": raw, "concepts": raw}
        if isinstance(raw, dict):
            # Unify alternate key names
            if not raw.get("nodes") and raw.get("concepts"):
                raw["nodes"] = raw["concepts"]
            elif not raw.get("concepts") and raw.get("nodes"):
                raw["concepts"] = raw["nodes"]
            if not raw.get("nodes") and not raw.get("concepts"):
                raise ValueError(
                    f"ConceptGraphSchema payload is missing 'nodes' or 'concepts': {raw}"
                )
            raw.setdefault("topic", hint or "DSA")
        return cls.model_validate(raw)


# ---------------------------------------------------------------------------
# Learning content
# ---------------------------------------------------------------------------

class CheckpointQuestion(BaseModel):
    """A 4-option MCQ checkpoint embedded in learning content."""
    question: str = Field(..., description="Direct, concept-specific checkpoint question")
    options: List[str] = Field(..., min_length=4, max_length=4, description="Exactly 4 options")
    correct_option: str = Field(..., description="Exact matching text of the correct choice")
    explanation: str = Field(..., description="Why this option is correct")

    @staticmethod
    def _clean_single_option(opt: Any) -> str:
        if opt is None:
            return ""
        if isinstance(opt, dict):
            txt = opt.get("text") or opt.get("label") or opt.get("option") or opt.get("id") or ""
            return str(txt).strip()
        s = str(opt).strip()
        if (s.startswith("{") and s.endswith("}")) or ("'text':" in s) or ('"text":' in s):
            try:
                import ast
                parsed = ast.literal_eval(s)
                if isinstance(parsed, dict):
                    txt = parsed.get("text") or parsed.get("label") or parsed.get("option") or parsed.get("id") or ""
                    return str(txt).strip()
            except Exception:
                pass
            try:
                import json
                parsed = json.loads(s)
                if isinstance(parsed, dict):
                    txt = parsed.get("text") or parsed.get("label") or parsed.get("option") or parsed.get("id") or ""
                    return str(txt).strip()
            except Exception:
                pass
            import re
            m = re.search(r"['\"]text['\"]\s*:\s*['\"]((?:[^'\\]|\\.)*)['\"]", s)
            if m:
                return m.group(1).replace("\\'", "'").replace('\\"', '"').strip()
        return s

    @model_validator(mode="before")
    @classmethod
    def _preprocess_checkpoint(cls, data: Any) -> Any:
        if isinstance(data, dict):
            opts = data.get("options")
            if isinstance(opts, list):
                cleaned = [cls._clean_single_option(o) for o in opts if cls._clean_single_option(o)]
                data["options"] = cleaned
            corr = data.get("correct_option") or data.get("correct_option_id")
            if corr:
                data["correct_option"] = cls._clean_single_option(corr)
        return data

    @model_validator(mode="after")
    def _correct_in_options(self) -> "CheckpointQuestion":
        if not self.options or not isinstance(self.options, list):
            self.options = ["Option A", "Option B", "Option C", "Option D"]
        opts_list = [self._clean_single_option(o) for o in self.options if self._clean_single_option(o)]
        while len(opts_list) < 4:
            opts_list.append(f"Option {chr(65 + len(opts_list))}")
        self.options = opts_list[:4]

        # 1. Exact match
        corr_clean = self._clean_single_option(self.correct_option)
        if corr_clean in self.options:
            self.correct_option = corr_clean
            return self

        # 2. Fuzzy match (strip whitespace)
        if not corr_clean:
            self.correct_option = self.options[0]
            return self

        for idx, opt in enumerate(self.options):
            if corr_clean == opt.strip():
                self.correct_option = self.options[idx]
                return self

        # 3. Match letter choice like "A", "B", "C", "D", "A.", "A)", "Option A"
        letter_map = {"A": 0, "B": 1, "C": 2, "D": 3}
        clean_upper = corr_clean.upper().replace("OPTION", "").replace(".", "").replace(")", "").replace(":", "").strip()
        if clean_upper in letter_map:
            idx = letter_map[clean_upper]
            if 0 <= idx < len(self.options):
                self.correct_option = self.options[idx]
                return self

        # 4. Check if correct_option is contained within an option or vice versa
        for idx, opt in enumerate(self.options):
            if corr_clean.lower() in opt.lower() or opt.lower().startswith(corr_clean.lower()):
                self.correct_option = self.options[idx]
                return self

        # 5. Last resort fallback: default to first option instead of raising Exception
        self.correct_option = self.options[0]
        return self


class Complexity(BaseModel):
    """Time/space complexity breakdown for a DSA concept."""
    best_case: str = Field(..., description="e.g. O(1)")
    worst_case: str = Field(..., description="e.g. O(n)")
    space: str = Field(..., description="Auxiliary space, e.g. O(1)")
    explanation: str = Field(..., description="Brief mechanical reason for the complexity")


# Backward-compat alias
ComplexityBreakdown = Complexity


class LessonContentSchema(StructuredOutputSchema):
    """Structured lesson content for a single DSA concept node."""
    concept_id: str
    intuition: str = Field(..., description="One-sentence 'why does this exist?' hook")
    core_idea: str = Field(..., description="The key insight in 2-3 sentences")
    mechanism: str = Field(..., description="Step-by-step operational description")
    step_by_step_example: str = Field(..., description="Concrete trace on a small example")
    code_snippet: str = Field(..., description="Production-grade executable code")
    language: str = Field(default="python", description="Target programming language")
    complexity: Complexity
    edge_cases: List[str] = Field(default_factory=list)
    common_mistakes: List[str] = Field(default_factory=list)
    checkpoint: CheckpointQuestion

    _format_example: ClassVar[str] = """{
  "concept_id": "array_two_pointer",
  "intuition": "Two pointers converge to reduce search space from O(n²) to O(n).",
  "core_idea": "Place one pointer at the start and one at the end...",
  "mechanism": "1. Sort the array. 2. Initialize left=0, right=n-1...",
  "step_by_step_example": "Given [2, 7, 11, 15], target=9: left=0(2), right=3(15)...",
  "code_snippet": "def two_sum(nums, target):\\n    ...",
  "language": "python",
  "complexity": {"best_case": "O(n)", "worst_case": "O(n)", "space": "O(1)", "explanation": "Single pass with two pointers."},
  "edge_cases": ["empty array", "single element", "all duplicates"],
  "common_mistakes": ["forgetting to sort first", "off-by-one on pointer bounds"],
  "checkpoint": {"question": "What happens if...", "options": ["A","B","C","D"], "correct_option": "B", "explanation": "Because..."}
}"""

    @classmethod
    def normalize(cls, raw: Any, hint: str = "") -> "LessonContentSchema":
        if not isinstance(raw, dict):
            raw = {}

        cid = raw.get("concept_id") or hint or "concept"
        raw["concept_id"] = str(cid).strip().lower().replace(" ", "_")

        # Fill missing top-level text fields
        intuition = raw.get("intuition") or raw.get("core_idea") or f"Core intuition and practical motivation for {cid}."
        core_idea = raw.get("core_idea") or raw.get("intuition") or f"Key structural principle and operational idea behind {cid}."
        mechanism = raw.get("mechanism") or f"1. Initialize data layout for {cid}.\n2. Process elements according to rules.\n3. Return result."
        step_by_step = raw.get("step_by_step_example") or f"Walkthrough example for {cid}."
        code = raw.get("code_snippet") or f"# Implementation example for {cid}\npass"

        raw["intuition"] = str(intuition)
        raw["core_idea"] = str(core_idea)
        raw["mechanism"] = str(mechanism).replace("\\n", "\n").replace('\\"', '"')
        raw["step_by_step_example"] = str(step_by_step).replace("\\n", "\n").replace('\\"', '"')
        raw["code_snippet"] = str(code).replace("\\n", "\n").replace("\\t", "  ").replace('\\"', '"')
        raw.setdefault("language", "python")

        # Normalize complexity
        comp = raw.get("complexity")
        if not isinstance(comp, dict):
            comp = {}
        raw["complexity"] = {
            "best_case": str(comp.get("best_case") or "O(1)"),
            "worst_case": str(comp.get("worst_case") or "O(n)"),
            "space": str(comp.get("space") or "O(1)"),
            "explanation": str(comp.get("explanation") or "Standard operation complexity analysis.")
        }

        # Normalize lists
        if not isinstance(raw.get("edge_cases"), list) or not raw.get("edge_cases"):
            raw["edge_cases"] = ["empty input", "single element"]
        if not isinstance(raw.get("common_mistakes"), list) or not raw.get("common_mistakes"):
            raw["common_mistakes"] = ["off-by-one boundary errors"]

        # Normalize checkpoint
        cp = raw.get("checkpoint")
        if not isinstance(cp, dict):
            cp = {}
        
        opts = cp.get("options")
        if isinstance(opts, list):
            clean_opts = [CheckpointQuestion._clean_single_option(o) for o in opts if CheckpointQuestion._clean_single_option(o)]
            while len(clean_opts) < 4:
                clean_opts.append(f"Option {chr(65 + len(clean_opts))}")
            cp["options"] = clean_opts[:4]
        else:
            cp["options"] = ["Option A", "Option B", "Option C", "Option D"]

        if not cp.get("question"):
            cp["question"] = f"What is the key structural principle of {cid}?"
        if not cp.get("explanation"):
            cp["explanation"] = "Validates core mechanism and boundary conditions."

        corr = cp.get("correct_option") or cp.get("correct_option_id")
        if not corr:
            cp["correct_option"] = cp["options"][0]
        else:
            cp["correct_option"] = CheckpointQuestion._clean_single_option(corr)

        raw["checkpoint"] = cp

        return cls.model_validate(raw)


class LessonOverviewSchema(StructuredOutputSchema):
    """Part 1: Core lesson explanation and executable code snippet."""
    concept_id: str
    intuition: str = Field(..., description="One-sentence 'why does this exist?' hook")
    core_idea: str = Field(..., description="The key insight in 2-3 sentences")
    mechanism: str = Field(..., description="Step-by-step operational description")
    code_snippet: str = Field(..., description="Production-grade compilable executable code")
    language: str = Field(default="python", description="Target programming language")

    _format_example: ClassVar[str] = """{
  "concept_id": "array_prefix_sum",
  "intuition": "Prefix sums reduce range sum queries from O(n) to O(1).",
  "core_idea": "Build a cumulative sum array P where P[i] is the sum up to index i.",
  "mechanism": "1. Set P[0] = arr[0]. 2. Loop i from 1..n: P[i] = P[i-1] + arr[i].",
  "code_snippet": "def build_prefix_sum(arr):...",
  "language": "python"
}"""

    @classmethod
    def normalize(cls, raw: Any, hint: str = "") -> "LessonOverviewSchema":
        if not isinstance(raw, dict):
            raw = {}
        cid = raw.get("concept_id") or hint or "concept"
        raw["concept_id"] = str(cid).strip().lower().replace(" ", "_")
        raw["intuition"] = str(raw.get("intuition") or raw.get("core_idea") or f"Core intuition for {cid}.")
        raw["core_idea"] = str(raw.get("core_idea") or raw.get("intuition") or f"Key structural principle for {cid}.")
        raw["mechanism"] = str(raw.get("mechanism") or f"1. Process {cid} elements.").replace("\\n", "\n").replace('\\"', '"')
        raw["code_snippet"] = str(raw.get("code_snippet") or f"# Code for {cid}\npass").replace("\\n", "\n").replace("\\t", "  ").replace('\\"', '"')
        raw.setdefault("language", "python")
        return cls.model_validate(raw)


class LessonDeepDiveSchema(StructuredOutputSchema):
    """Part 2: Deep-dive walkthrough, complexity, edge cases, and checkpoint question."""
    concept_id: str
    step_by_step_example: str = Field(..., description="Concrete trace on a small example")
    complexity: Complexity
    edge_cases: List[str] = Field(default_factory=list)
    common_mistakes: List[str] = Field(default_factory=list)
    checkpoint: CheckpointQuestion

    _format_example: ClassVar[str] = """{
  "concept_id": "array_prefix_sum",
  "step_by_step_example": "Given arr=[3,1,4]: Step 1 P[0]=3, Step 2 P[1]=4...",
  "complexity": {"best_case": "O(n)", "worst_case": "O(n)", "space": "O(n)", "explanation": "Single pass with array allocation."},
  "edge_cases": ["empty array", "single element"],
  "common_mistakes": ["forgetting to handle L=0"],
  "checkpoint": {"question": "What is the formula?", "options": ["P[R]-P[L-1]", "P[R]", "P[L]", "None"], "correct_option": "P[R]-P[L-1]", "explanation": "Removes sum prior to L."}
}"""

    @classmethod
    def normalize(cls, raw: Any, hint: str = "") -> "LessonDeepDiveSchema":
        if not isinstance(raw, dict):
            raw = {}
        cid = raw.get("concept_id") or hint or "concept"
        raw["concept_id"] = str(cid).strip().lower().replace(" ", "_")
        raw["step_by_step_example"] = str(raw.get("step_by_step_example") or f"Walkthrough example for {cid}.").replace("\\n", "\n").replace('\\"', '"')

        comp = raw.get("complexity")
        if not isinstance(comp, dict):
            comp = {}
        raw["complexity"] = {
            "best_case": str(comp.get("best_case") or "O(1)"),
            "worst_case": str(comp.get("worst_case") or "O(n)"),
            "space": str(comp.get("space") or "O(1)"),
            "explanation": str(comp.get("explanation") or "Standard operation complexity analysis.")
        }

        if not isinstance(raw.get("edge_cases"), list) or not raw.get("edge_cases"):
            raw["edge_cases"] = ["empty input", "single element"]
        if not isinstance(raw.get("common_mistakes"), list) or not raw.get("common_mistakes"):
            raw["common_mistakes"] = ["off-by-one boundary errors"]

        cp = raw.get("checkpoint")
        if not isinstance(cp, dict):
            cp = {}
        opts = cp.get("options")
        if isinstance(opts, list):
            clean_opts = [CheckpointQuestion._clean_single_option(o) for o in opts if CheckpointQuestion._clean_single_option(o)]
            while len(clean_opts) < 4:
                clean_opts.append(f"Option {chr(65 + len(clean_opts))}")
            cp["options"] = clean_opts[:4]
        else:
            cp["options"] = ["Option A", "Option B", "Option C", "Option D"]

        if not cp.get("question"):
            cp["question"] = f"What is the key structural principle of {cid}?"
        if not cp.get("explanation"):
            cp["explanation"] = "Validates core mechanism and boundary conditions."

        corr = cp.get("correct_option") or cp.get("correct_option_id")
        if not corr:
            cp["correct_option"] = cp["options"][0]
        else:
            cp["correct_option"] = CheckpointQuestion._clean_single_option(corr)

        raw["checkpoint"] = cp
        return cls.model_validate(raw)


class LessonExplanationSchema(StructuredOutputSchema):
    """Re-teach lesson using a specific pedagogical strategy."""
    concept_id: str
    strategy: Literal["analogy", "visual_diagram", "interactive_exercise", "worked_example"]
    content: str = Field(..., min_length=20)

    _format_example: ClassVar[str] = (
        '{"concept_id": "stack_push_pop", "strategy": "analogy", '
        '"content": "Think of a stack like a pile of plates..."}'
    )

    @classmethod
    def normalize(cls, raw: Any, hint: str = "") -> "LessonExplanationSchema":
        if isinstance(raw, dict):
            raw.setdefault("concept_id", hint or "concept")
        return cls.model_validate(raw)


# ===========================================================================
# 3. Questions
# ===========================================================================

class GeneratedTestCaseItem(BaseModel):
    """One test case for a practice / coding question."""
    input: str = Field(
        ...,
        description=(
            "Raw JSON value only — NO variable assignment. "
            "Good: '[1, 2, 3]'  Bad: 'nums = [1, 2, 3]'"
        ),
    )
    expected_output: str = Field(..., description="Exact return value as a string")

    @field_validator("input", "expected_output", mode="before")
    @classmethod
    def _stringify_value(cls, v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, (dict, list)):
            return json.dumps(v)
        return str(v)

    @field_validator("input")
    @classmethod
    def _reject_assignment(cls, v: str) -> str:
        v = str(v).strip()
        if "=" in v and not v.startswith(("[", "{")):
            parts = v.split("=", 1)
            if len(parts) == 2 and parts[1].strip():
                return parts[1].strip()
        return v



class GeneratedQuestionItem(BaseModel):
    """A single generated assessment question."""
    type: Literal["mcq", "true_false", "short_answer", "open_ended", "practice"]
    concept_id: str
    pattern: Optional[str] = None
    difficulty: int = Field(..., ge=1, le=5, description="1=Easy … 5=Hard (LeetCode scale)")
    question: str = Field(..., min_length=5)

    # MCQ / true_false fields
    options: Optional[List[str]] = None
    correct_option: Optional[str] = None

    # Open-ended / practice fields
    rubric: Optional[str] = None
    expected_time_complexity: Optional[str] = None
    expected_space_complexity: Optional[str] = None
    function_signature: Optional[str] = None
    test_cases: Optional[List[GeneratedTestCaseItem]] = None
    expected_answer: Optional[str] = None

    execution_method: Literal["llm_rubric", "browser_compiler", "unit_test"] = "llm_rubric"

    @model_validator(mode="after")
    def _validate_type_requirements(self) -> "GeneratedQuestionItem":
        if self.type in ("mcq", "true_false"):
            if not self.options:
                raise ValueError(f"Question type '{self.type}' requires 'options'.")
            if not self.correct_option:
                raise ValueError(f"Question type '{self.type}' requires 'correct_option'.")
        if self.type in ("open_ended", "practice"):
            if not self.rubric:
                raise ValueError(f"Question type '{self.type}' requires 'rubric'.")
        return self

    @field_validator("difficulty", mode="before")
    @classmethod
    def _clamp_difficulty(cls, v: Any) -> int:
        try:
            return max(1, min(5, int(v)))
        except (TypeError, ValueError):
            return 3


class QuestionBatchSchema(StructuredOutputSchema):
    """Batch of generated questions."""
    questions: List[GeneratedQuestionItem] = Field(..., min_length=1)

    _format_example: ClassVar[str] = (
        '{"questions": [{"type": "mcq", "concept_id": "stack_definition", "difficulty": 2, '
        '"question": "What is the primary characteristic of a stack data structure?", '
        '"options": ["Last-In, First-Out (LIFO) access", "First-In, First-Out (FIFO) access", "Random access by index", "Hierarchical tree structure"], '
        '"correct_option": "Last-In, First-Out (LIFO) access"}]}'
    )

    @classmethod
    def normalize(cls, raw: Any, hint: str = "") -> "QuestionBatchSchema":
        # If LLM returned a bare list of questions
        if isinstance(raw, list):
            raw = {"questions": raw}

        if isinstance(raw, dict):
            # Try alternate key names
            if not isinstance(raw.get("questions"), list):
                for alt_key in ("questions_list", "items", "data", "results", "question_batch", "qs"):
                    if isinstance(raw.get(alt_key), list):
                        raw["questions"] = raw[alt_key]
                        break
            # Single question wrapped in dict
            if not isinstance(raw.get("questions"), list):
                if "question" in raw:
                    raw = {"questions": [raw]}
                else:
                    raw["questions"] = []

            # Patch individual question defaults
            if isinstance(raw.get("questions"), list):
                concept_fallback = hint or "general"
                import re
                for q in raw["questions"]:
                    if isinstance(q, dict):
                        q.setdefault("concept_id", concept_fallback)
                        q.setdefault("type", "mcq")
                        try:
                            q["difficulty"] = max(1, min(5, int(q.get("difficulty", 3))))
                        except (TypeError, ValueError):
                            q["difficulty"] = 3

                        # Ensure 'question' field is present and valid string
                        if not q.get("question") or len(str(q.get("question")).strip()) < 3:
                            if q.get("rubric"):
                                q["question"] = str(q["rubric"])
                            elif q.get("function_signature"):
                                q["question"] = f"Implement problem: {q['function_signature']}"
                            else:
                                q["question"] = f"Solve the challenge for {q['concept_id']}"

                        # Sanitize MCQ / true_false requirements & parse embedded options if present
                        if q.get("type") in ("mcq", "true_false"):
                            raw_q_text = str(q.get("question", "")).strip()
                            raw_opts = q.get("options")
                            if not isinstance(raw_opts, list):
                                raw_opts = []
                            raw_corr = str(q.get("correct_option") or "").strip()

                            # Check for embedded options in question text
                            extracted = []
                            clean_lines = []
                            # Line-by-line check
                            opt_line_pattern = re.compile(r"^\s*(?:([A-D])|([1-4]))[\)\.\:\-]\s*(.+)$", re.IGNORECASE)
                            for line in raw_q_text.split("\n"):
                                m = opt_line_pattern.match(line)
                                if m:
                                    let = (m.group(1) or "").upper()
                                    dig = m.group(2) or ""
                                    key = let if let else ("ABCD"[int(dig) - 1] if dig and 1 <= int(dig) <= 4 else "")
                                    extracted.append((key, m.group(3).strip()))
                                else:
                                    clean_lines.append(line)

                            if len(extracted) < 2:
                                # Check inline options
                                inline_parts = re.split(r"\s+(?=(?:[A-D]|[1-4])[\)\.\:\-]\s+)", raw_q_text)
                                if len(inline_parts) >= 3:
                                    temp_extracted = []
                                    for part in inline_parts[1:]:
                                        m = opt_line_pattern.match(part.strip())
                                        if m:
                                            let = (m.group(1) or "").upper()
                                            dig = m.group(2) or ""
                                            key = let if let else ("ABCD"[int(dig) - 1] if dig and 1 <= int(dig) <= 4 else "")
                                            temp_extracted.append((key, m.group(3).strip()))
                                    if len(temp_extracted) >= 2:
                                        extracted = temp_extracted
                                        clean_lines = [inline_parts[0].strip()]

                            are_raw_opts_letters = not raw_opts or all(
                                len(str(o).strip()) <= 2 and (str(o).strip().upper() in ["A", "B", "C", "D"] or str(o).strip() in ["1", "2", "3", "4"])
                                for o in raw_opts
                            )

                            if extracted and (are_raw_opts_letters or len(raw_opts) < len(extracted)):
                                q["question"] = "\n".join(clean_lines).strip()
                                q["options"] = [text for _, text in extracted]
                                # Map correct_option if it was letter or index
                                resolved_corr = raw_corr
                                for key, text in extracted:
                                    if raw_corr.upper() == key or raw_corr == text or raw_corr.upper().startswith(key):
                                        resolved_corr = text
                                        break
                                q["correct_option"] = resolved_corr
                            else:
                                if len(raw_opts) == 0:
                                    q["options"] = ["Option A", "Option B", "Option C", "Option D"]
                                if not q.get("correct_option"):
                                    q["correct_option"] = q["options"][0]

                        # Ensure practice / open_ended requirements
                        if q.get("type") in ("open_ended", "practice"):
                            if not q.get("rubric"):
                                q["rubric"] = f"Evaluate implementation correctness for {q['concept_id']}"

        return cls.model_validate(raw)


class QuestionTextSchema(StructuredOutputSchema):
    """Fast, small structured payload for a single question.

    Deliberately excludes test cases so the question text can be produced in
    one short LLM round-trip. Test cases are generated by a dedicated tool
    (``TestCaseListSchema``) once the statement and signature are known.
    """
    type: Literal["mcq", "true_false", "short_answer", "open_ended", "practice"]
    concept_id: str
    difficulty: int = Field(..., ge=1, le=5)
    question: str = Field(..., min_length=5)
    options: Optional[List[str]] = None
    correct_option: Optional[str] = None
    rubric: Optional[str] = None
    expected_time_complexity: Optional[str] = None
    expected_space_complexity: Optional[str] = None
    function_signature: Optional[str] = None
    execution_mode: Optional[str] = None
    input_adapter: Optional[str] = None
    output_adapter: Optional[str] = None
    supported_languages: Optional[List[str]] = None
    test_case_schema: Optional[str] = None
    comparison_mode: Optional[str] = None
    problem_version: Optional[int] = None
    absolute_tolerance: Optional[float] = None
    relative_tolerance: Optional[float] = None
    # Complete, self-contained starter code (imports + any helper/node class
    # definitions + the function stub). Generated by the model so the harness
    # never has to guess structural definitions.
    starter_code: Optional[str] = None
    expected_answer: Optional[str] = None
    explanation: Optional[str] = None

    _format_example: ClassVar[str] = (
        '{"type": "practice", "concept_id": "two_sum", "difficulty": 3, '
        '"question": "Given an array of integers `nums` and an integer `target`, return indices of the two numbers that add up to `target`.", '
        '"function_signature": "def two_sum(nums: list[int], target: int) -> list[int]:", '
        '"starter_code": "from typing import List\\n\\ndef two_sum(nums: List[int], target: int) -> List[int]:\\n    pass", '
        '"rubric": "O(n) hash-map solution expected.", "expected_time_complexity": "O(n)", '
        '"expected_space_complexity": "O(n)"}'
    )

    @field_validator("difficulty", mode="before")
    @classmethod
    def _clamp_difficulty(cls, v: Any) -> int:
        try:
            return max(1, min(5, int(v)))
        except (TypeError, ValueError):
            return 3

    @model_validator(mode="after")
    def _validate(self) -> "QuestionTextSchema":
        if self.type == "practice" and self.function_signature:
            self.execution_mode = self.execution_mode or "function"
            self.input_adapter = self.input_adapter or "json_args_v1"
            self.output_adapter = self.output_adapter or "json_value_v1"
            self.supported_languages = self.supported_languages or ["python"]
            self.test_case_schema = self.test_case_schema or "args_kwargs_json_v1"
            self.comparison_mode = self.comparison_mode or "exact_json_v1"
            self.problem_version = self.problem_version or 2
        if self.type in ("mcq", "true_false"):
            if not self.options:
                raise ValueError("mcq/true_false requires options")
            opts = [str(o).strip() for o in self.options if str(o).strip()]
            self.options = opts
            if not self.correct_option or self.correct_option not in opts:
                self.correct_option = opts[0]
        return self

    @classmethod
    def normalize(cls, raw: Any, hint: str = "") -> "QuestionTextSchema":
        if isinstance(raw, list):
            raw = raw[0] if raw else {}
        if isinstance(raw, dict):
            if isinstance(raw.get("questions"), list) and raw["questions"]:
                raw = raw["questions"][0]
            raw.setdefault("concept_id", hint or "general")
            raw.setdefault("type", "mcq")
            raw.setdefault("difficulty", 3)
            if not raw.get("question"):
                raw["question"] = raw.get("prompt") or raw.get("title") or f"Practice {hint}"
            if raw.get("type") in ("mcq", "true_false"):
                opts = raw.get("options")
                if not isinstance(opts, list) or len(opts) < 2:
                    raw["options"] = ["Option A", "Option B", "Option C", "Option D"]
                if not raw.get("correct_option"):
                    raw["correct_option"] = raw["options"][0]
        return cls.model_validate(raw)


class TestCaseListSchema(StructuredOutputSchema):
    """Dedicated output for the test-case generation tool."""
    test_cases: List[GeneratedTestCaseItem] = Field(..., min_length=1)

    _format_example: ClassVar[str] = (
        '{"test_cases": ['
        '{"input": "[2, 7, 11, 15], 9", "expected_output": "[0, 1]"}, '
        '{"input": "[3, 2, 4], 6", "expected_output": "[1, 2]"}, '
        '{"input": "[3, 3], 6", "expected_output": "[0, 1]"}]}'
    )

    @classmethod
    def normalize(cls, raw: Any, hint: str = "") -> "TestCaseListSchema":
        if isinstance(raw, list):
            raw = {"test_cases": raw}
        if isinstance(raw, dict):
            if not isinstance(raw.get("test_cases"), list):
                for key in ("cases", "tests", "results", "data", "testcases"):
                    if isinstance(raw.get(key), list):
                        raw["test_cases"] = raw[key]
                        break
            if not isinstance(raw.get("test_cases"), list):
                raw["test_cases"] = []
            cleaned = []
            for tc in raw["test_cases"]:
                if not isinstance(tc, dict):
                    continue
                inp = tc.get("input", tc.get("args", ""))
                out = tc.get("expected_output", tc.get("output", tc.get("expected", "")))
                if isinstance(inp, (list, dict)):
                    inp = json.dumps(inp)
                if isinstance(out, (list, dict)):
                    out = json.dumps(out)
                if str(inp).strip() == "" and str(out).strip() == "":
                    continue
                cleaned.append({"input": str(inp).strip(), "expected_output": str(out).strip()})
            raw["test_cases"] = cleaned
        return cls.model_validate(raw)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

class RubricEvaluationSchema(StructuredOutputSchema):
    """Multi-signal evaluation result for an open-ended or practice submission."""
    llm_quality_score: float = Field(..., ge=0.0, le=1.0)
    correctness_score: float = Field(default=0.0, ge=0.0, le=1.0)
    complexity_score: float = Field(default=0.0, ge=0.0, le=1.0)
    code_quality_score: float = Field(default=0.0, ge=0.0, le=1.0)
    edge_case_score: float = Field(default=0.0, ge=0.0, le=1.0)
    feedback: str = Field(..., min_length=5)
    conceptual_breakdown: Dict[str, Any] = Field(default_factory=dict)

    _format_example: ClassVar[str] = (
        '{"llm_quality_score": 0.82, "correctness_score": 0.9, '
        '"complexity_score": 0.7, "code_quality_score": 0.8, '
        '"edge_case_score": 0.6, "feedback": "Correct approach but...", '
        '"conceptual_breakdown": {}}'
    )


# ===========================================================================
# 4. Re-teach
# ===========================================================================

class ReteachLessonSchema(StructuredOutputSchema):
    """Adaptive re-teach content using a new pedagogical strategy."""
    concept_id: str
    strategy: Literal["analogy", "visual_diagram", "interactive_exercise", "worked_example"]
    content: str = Field(..., min_length=30)
    complexity_explanation: Optional[str] = Field(
        default=None,
        description="Explicit 'why is this the complexity?' explanation"
    )

    _format_example: ClassVar[str] = (
        '{"concept_id": "stack_push_pop", "strategy": "visual_diagram", '
        '"content": "Step 1: Start with empty stack []\\nStep 2: push(10)...", '
        '"complexity_explanation": "Push/pop are O(1) because..."}'
    )

    @classmethod
    def normalize(cls, raw: Any, hint: str = "") -> "ReteachLessonSchema":
        if isinstance(raw, dict):
            raw.setdefault("concept_id", hint or "concept")
        return cls.model_validate(raw)
