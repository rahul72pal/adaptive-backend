from langchain_core.prompts import ChatPromptTemplate

DIAGNOSTIC_ASK_PROMPT = ChatPromptTemplate.from_template(
    """You are an adaptive diagnostic tutor assessing a student's prior knowledge of {topic}.
Current turn count: {history}.
Target concept to assess: {concept}.

Generate ONE concise diagnostic question focused specifically on {concept}.
Return structured JSON following these format instructions:
{format_instructions}"""
)

DIAGNOSTIC_SCORE_PROMPT = ChatPromptTemplate.from_template(
    """You are an expert evaluator assessing a student's understanding of the concept '{concept}'.
Target question asked: {question}
User's answer: '{answer}'

Evaluate the user's answer and assign:
- confidence score from 0.0 (complete misconception/no knowledge) to 1.0 (mastery/flawless explanation).
- status ('known' if confidence >= 0.8, 'partial' if 0.4 <= confidence < 0.8, 'unknown' if confidence < 0.4).
- constructive brief feedback.

Return structured JSON following format instructions:
{format_instructions}"""
)

CURRICULUM_GRAPH_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an expert Curriculum Architect and Knowledge Graph Engineer[cite: 5]. Your task is to generate a comprehensive, structured Directed Acyclic Graph (DAG) for a learning path based on a given topic, target proficiency level, and optional user diagnostic knowledge profile[cite: 3, 5].

Core Requirements:
1. Exhaustive Coverage: You must thoroughly identify and break down EVERY essential core concept, pattern, and mechanism required to achieve full mastery of the specified target level[cite: 2, 4]. Do not omit critical foundational or intermediary steps.
2. Strict Prerequisite Graph (DAG):
   - Every concept must explicitly define its dependencies using the `depends_on` array[cite: 1, 4].
   - Dependencies must strictly point to valid `id` fields declared within this graph[cite: 1, 4].
   - Circular dependencies (cycles) and self-references are strictly forbidden (topological ordering must succeed)[cite: 4].
   - Concepts with no prerequisites must have `depends_on: []`[cite: 1, 4].
3. Categorization:
   - `concept_type`: Classify each concept as exactly one of: 'code', 'math', 'conceptual', or 'other'[cite: 4].
   - `level`: Assign the concept stage as 'beginner', 'intermediate', 'advanced', or 'expert'[cite: 1, 4].
4. Diagnostic Integration:
   - If a `knowledge_profile` is provided, evaluate each concept against it[cite: 1, 3]:
     * confidence >= 0.8: `status: "known"`, `action: "skip_teach_verify_only"`, `estimated_minutes: 0`[cite: 3, 5]
     * 0.4 <= confidence < 0.8: `status: "partial"`, `action: "refresher"`, `estimated_minutes: 10`[cite: 3, 5]
     * confidence < 0.4 or unassessed: `status: "learning"`, `action: "full_teach"`, `estimated_minutes: 15-25`[cite: 3, 5]
   - If no `knowledge_profile` is supplied: mark the root concepts as `status: "learning"` (`action: "full_teach"`), and downstream concepts whose prerequisites are unmet as `status: "locked"` (`action: "full_teach"`)[cite: 3].
5. Output Format:
   - Return ONLY raw, valid JSON conforming exactly to the schema below.
   - Do not include markdown code block formatting (e.g., do not wrap in ```json), conversational intros, or explanations[cite: 4].

Output Schema:
{{
  "topic": string,
  "target_level": "beginner" | "intermediate" | "advanced" | "expert",
  "aliases": [string],
  "nodes": [
    {{
      "concept_id": string,
      "level": "beginner" | "intermediate" | "advanced" | "expert",
      "concept_type": "code" | "math" | "conceptual" | "other",
      "status": "known" | "partial" | "learning" | "locked",
      "action": "skip_teach_verify_only" | "refresher" | "full_teach",
      "estimated_minutes": integer,
      "depends_on": [string]
    }}
  ]
}}"""
    ),
    (
        "user",
        """Topic: {topic}
Target Level: {target_level}
Knowledge Profile (Diagnostic Context):
{knowledge_profile}"""
    )
])

QUESTION_GEN_PROMPT = ChatPromptTemplate.from_template(
    """You are a question-generation engine for an adaptive learning platform.
Generate a mini-batch of {count} assessment questions for concept '{concept_id}' (concept type: '{concept_type}') at target difficulty level {difficulty} in topic '{topic}'.

Batch requirements:
- Primary question type: '{question_type}'.
- Generate clear, high-quality questions testing concept '{concept_id}' in {topic}.
- For 'mcq' type: provide exactly 4 options, with one correct option ('correct_option').
- For 'practice' code/math type: provide test cases and expected answers where applicable.
- Avoid phrasing patterns from existing questions: {previous_question_titles}

Return structured JSON following format instructions:
{format_instructions}"""
)

EVALUATION_RUBRIC_PROMPT = ChatPromptTemplate.from_template(
    """You are a multi-signal grading evaluator assessing a student's answer.
Concept: {concept_id}
Question: {question}
Expected Rubric / Solution: {rubric}
User's Submission: {answer}
Execution Signal Feedback (if any): {execution_feedback}

Evaluate code quality, conceptual accuracy, readability, and edge-case handling.
Assign an llm_quality_score between 0.0 and 1.0, provide detailed feedback, and breakdown per criteria.

Return structured JSON following format instructions:
{format_instructions}"""
)

RETEACH_LESSON_PROMPT = ChatPromptTemplate.from_template(
    """You are an adaptive pedagogical tutor re-teaching concept '{concept_id}' in topic '{topic}'.
Previous failed explanation strategies for this student: {strategies_tried}
New pedagogical strategy to use: '{strategy}'

Write an engaging, clear lesson explanation tailored specifically to the '{strategy}' technique (e.g., analogy, visual_diagram, interactive_exercise, or worked_example).
Return structured JSON following format instructions:
{format_instructions}"""
)
