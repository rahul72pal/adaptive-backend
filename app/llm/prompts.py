"""
DSA-oriented prompt templates for the adaptive learning platform.

Key changes from the generic version:
  1. CURRICULUM_GRAPH_PROMPT no longer hardcodes array-specific examples —
     it was silently biasing every topic toward array pedagogy. Guidance is
     now general DSA rigor; the LLM derives topic-specific content from
     {topic} itself.
  2. Every concept node now carries a `pattern` tag (two_pointer,
     sliding_window, dfs, bfs, backtracking, dp_1d, dp_2d, greedy,
     binary_search, union_find, trie, heap, divide_conquer, ...) — this is
     the axis real interview-prep platforms (NeetCode, LeetCode patterns)
     organize by, and it's more useful than topic/level alone for a DSA
     platform specifically.
  3. Time/space complexity is now a first-class, explicitly graded signal
     everywhere — diagnostic scoring, question generation constraints, and
     evaluation rubric — not folded silently into "code quality."
  4. Question generation requires edge-case test coverage (empty input,
     single element, duplicates, negative numbers, large-N for complexity
     verification) rather than leaving edge cases to the LLM's discretion.
"""

from langchain_core.prompts import ChatPromptTemplate

# ---------------------------------------------------------------------------
# 1. DIAGNOSTIC — ask node
# ---------------------------------------------------------------------------

DIAGNOSTIC_ASK_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an expert DSA (Data Structures & Algorithms) interviewer probing a
student's real understanding of the concept: '{concept}', within the topic {topic}.

Your task is to generate ONE precise, highly targeted diagnostic question.

Question Guidelines:
1. Test Mechanism, Not Definition: Never ask "What is a {concept}?" or other
   textbook-recall questions. Ask about the underlying mechanism, memory
   layout, or operational behavior instead.
2. Always Probe Complexity: Where applicable, the question must require the
   student to state or reason about time and/or space complexity — this is
   the single strongest signal of real DSA understanding vs memorized
   definitions.
3. Prefer a Concrete Trace: Where possible, frame the question around a
   specific small example the student must reason through (e.g. "Given the
   array [5, 3, 8, 1], walk through how a two-pointer approach would find
   a pair summing to 9 — what's the time complexity?") rather than an
   abstract question with no data to anchor to.
4. Concise & Direct Answerability: The question must be answerable in 2-4
   precise sentences so capability can be accurately evaluated without the
   student needing to write full code.
5. Do Not Test Adjacent Concepts: Stay strictly scoped to '{concept}' —
   don't require knowledge of prerequisite or downstream concepts to answer.

CRITICAL FORMAT REQUIREMENT: Respond ONLY with a valid JSON object matching the format instructions. Do NOT include any intro text, reasoning prose, or conversational commentary outside the JSON.

Return structured JSON following format instructions:
{format_instructions}"""
    ),
    (
        "user",
        """Topic: {topic}
Target concept: {concept}
Turn: {history}"""
    )
])

# ---------------------------------------------------------------------------
# 2. DIAGNOSTIC — score node
# ---------------------------------------------------------------------------

DIAGNOSTIC_SCORE_PROMPT = ChatPromptTemplate.from_template(
    """You are an expert DSA interviewer assessing a student's answer for the concept '{concept}'.
Question asked: {question}
Student's answer: '{answer}'

Grade against these specific signals, not general impression:
1. Correctness of the core mechanism/trace described.
2. Correctness of any time/space complexity claim made or implied — a
   student who gets the mechanism right but states the wrong Big-O should
   NOT score as "known."
3. Awareness of edge cases or trade-offs, if the question invited them.
4. Precision of language (vague hand-waving like "it's fast" without
   justification should score lower than a specific, reasoned answer).

Assign:
- confidence score from 0.0 (complete misconception/no knowledge) to 1.0
  (mechanism + complexity both correct and precisely reasoned).
- status ('known' if confidence >= 0.8, 'partial' if 0.4 <= confidence < 0.8,
  'unassessed' if unprobed, 'unknown' if confidence < 0.4).
- constructive brief feedback, specifically naming whether the gap (if any)
  was in mechanism understanding or complexity reasoning.

Return structured JSON following format instructions:
{format_instructions}"""
)

# ---------------------------------------------------------------------------
# 3. CURRICULUM GRAPH — generalized, no hardcoded array examples
# ---------------------------------------------------------------------------

CURRICULUM_GRAPH_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are an expert DSA Curriculum Architect, Knowledge Graph Engineer,
and Adaptive Learning Designer.

Your task is to generate a topic-specific learning graph for a
Data Structures & Algorithms adaptive learning platform.

The requested topic is provided dynamically as:

TOPIC: {topic}

The requested learning level is:

LEVEL: {target_level}

The graph must be generated specifically for the requested topic.
DO NOT assume the topic is an array, linked list, tree, graph,
dynamic programming, or any other specific DSA topic.

You must infer the actual knowledge structure, algorithms, patterns,
problem families, implementation techniques, and complexity concepts
that are relevant to the requested topic.

==================================================
1. CORE PRINCIPLE
==================================================

Generate a realistic DSA curriculum for:

Topic: {topic}
Level: {target_level}

The curriculum must be:

- topic-specific
- technically accurate
- progressively structured
- interview-oriented
- skill-oriented
- prerequisite-aware
- suitable for adaptive learning
- represented as a Directed Acyclic Graph (DAG)

Do not generate generic filler concepts merely to increase node count.

Every node must represent a meaningful, independently teachable or
assessable DSA skill.

==================================================
2. LEVEL-AWARE GENERATION
==================================================

Only generate concepts appropriate for the requested level.

The system generates levels progressively.

The available levels are:

- beginner
- intermediate
- advanced
- expert

When generating a level, use the previously completed levels as
context when provided.

Do NOT regenerate concepts that have already been mastered unless
they are required as a prerequisite for a new concept.

LEVEL GUIDANCE:

BEGINNER:
Focus on the foundational mental model and essential mechanics
required to understand the topic.

Possible areas include:

- fundamental terminology
- core representation
- basic operations
- basic algorithmic behavior
- simple implementation
- basic complexity
- fundamental patterns
- simple canonical problems

Do not blindly include all of these.
Only include what is actually relevant to {topic}.

INTERMEDIATE:
Focus on applying the topic to standard DSA problems.

Possible areas include:

- canonical problem-solving patterns
- common algorithmic techniques
- standard implementation patterns
- moderate complexity analysis
- combinations of previously learned concepts
- common interview problems

ADVANCED:
Focus on deeper algorithmic reasoning.

Possible areas include:

- advanced variants
- multiple-state reasoning
- optimization
- difficult edge cases
- combining multiple patterns
- advanced complexity analysis
- non-obvious problem transformations
- harder canonical problems

EXPERT:
Focus on advanced problem-solving and mastery.

Possible areas include:

- specialized algorithms
- advanced variants
- complex combinations
- optimization techniques
- difficult problem families
- recognizing hidden patterns
- designing solutions for unfamiliar problems
- trade-offs between approaches
- expert-level interview problems

These are guidelines, NOT mandatory categories.

The actual concepts must be determined from the topic.

==================================================
3. DO NOT USE FIXED NODE COUNTS
==================================================

Do NOT generate a fixed number of nodes.

The number of nodes must depend on:

- breadth of the topic
- complexity of the topic
- requested level
- number of meaningful sub-patterns
- number of prerequisite concepts
- number of independently assessable skills

Generate enough concepts for meaningful coverage, but avoid redundant
or artificial concepts.

A small topic may require only a few concepts.

A broad topic may require many concepts.

Quality and coverage are more important than a fixed number.

NEVER add generic filler concepts just to reach a target number.

==================================================
4. NODE TYPE
==================================================

Every node MUST have:

node_type:

- "learning"
- "practice"

Use:

learning
when the node primarily teaches a new concept, mechanism, algorithm,
pattern, strategy, or complexity idea.

Use:

practice
when the node primarily assesses the learner through problems,
questions, implementation tasks, traces, or application exercises.

A practice node should normally depend on one or more learning nodes.

A learning node should introduce or explain a meaningful concept.

Do not create practice nodes that require knowledge that has not
been introduced by their prerequisites.

A realistic progression may look like:

learning
→ practice
→ learning
→ practice
→ learning
→ practice

but this is NOT mandatory.

The structure should depend on the topic.

==================================================
5. PREREQUISITE DAG
==================================================

Every node MUST explicitly define:

depends_on

Dependencies must represent genuine learning prerequisites.

Rules:

- dependencies must reference valid node IDs
- no self-dependencies
- no circular dependencies
- graph must remain a valid DAG
- root nodes must have []
- advanced concepts should normally depend on relevant foundational
  concepts
- practice nodes should normally depend on the learning concepts
  required to solve them

Do not connect every node to the previous node.

The graph is NOT a simple linear sequence.

Multiple concepts may share prerequisites.

Example:

        foundational_concept
          /             \
         /               \
   learning_A         learning_B
       |                  |
   practice_A         practice_B
         \               /
          \             /
           advanced_concept

Create the dependency structure that actually reflects learning.

==================================================
6. EXCLUDED / PREVIOUS CONCEPT IDs & DUPLICATE RULES
==================================================

Previously Generated / Excluded Concept IDs:
{previous_level_concepts}

STRICT GRAPH GENERATION RULES:
1. Generate ONLY new concepts specifically appropriate for the {target_level} level of {topic}.
2. NEVER generate any node whose concept_id appears in Excluded Concept IDs.
3. Do NOT recreate or regenerate previous concepts.
4. Every concept node MUST have a unique, canonical concept_id (e.g., topic_concept_name).
5. Prerequisites (depends_on) can reference existing concept_ids from previous levels or newly generated concept_ids.
6. Generate only the number of nodes required for meaningful coverage. Hard limit: maximum 8 learning nodes and 8 practice nodes (total maximum 16 nodes per level). Do not create filler nodes to reach the limit.
7. Return ONLY a valid JSON object matching the requested schema without any markdown preambles or explanations.

==================================================
8. TOPIC-SPECIFIC PATTERNS
==================================================

Identify the actual algorithmic patterns relevant to {topic}.

Do not force a predefined pattern list onto every topic.

If a canonical DSA pattern applies, tag it.

Examples include:

- two_pointer
- sliding_window
- fast_slow_pointer
- dfs
- bfs
- backtracking
- dp_1d
- dp_2d
- greedy
- binary_search
- union_find
- trie
- heap
- divide_conquer
- topological_sort
- bit_manipulation
- prefix_sum

Use:

"none"

when no specific pattern applies.

If the requested topic requires another meaningful pattern not in this
list, use an appropriate descriptive pattern string rather than forcing
an incorrect existing category.

==================================================
9. COMPLEXITY
==================================================

Time and space complexity are first-class DSA learning signals.

Where relevant, concepts should teach:

- time complexity
- space complexity
- worst-case behavior
- average-case behavior
- amortized behavior
- trade-offs

Practice nodes involving implementation should be designed so that
complexity can be evaluated separately from correctness.

==================================================
10. TOPIC BOUNDARY
==================================================

Stay within the requested DSA topic.

Do not introduce unrelated domains such as:

- databases
- distributed systems
- operating systems
- frontend development
- system design

unless the requested topic explicitly requires them.

Prerequisite concepts from another DSA area may be referenced when
they are genuinely necessary.

==================================================
11. CONCEPT QUALITY
==================================================

Every node must satisfy ALL of the following:

1. It represents a real DSA skill or knowledge unit.
2. It can be taught independently.
3. It can be assessed independently.
4. It contributes to mastery of the requested topic.
5. Its prerequisite relationships are meaningful.
6. It is not merely a renamed version of another node.

Avoid generic filler such as:

- basic_operations
- common_patterns
- search_strategies
- traversal_mechanisms
- optimization_techniques
- real_world_applications

unless that exact concept is genuinely meaningful for the requested
topic.

==================================================
12. LEARNING → PRACTICE RELATIONSHIP
==================================================

A learning node should normally be followed or eventually connected
to one or more practice nodes.

Learning nodes answer:

"What does the learner need to understand?"

Practice nodes answer:

"Can the learner actually apply it?"

For example:

learning:
understand a specific algorithmic technique

practice:
solve a problem requiring that technique

Do not create practice nodes that simply repeat the definition.

==================================================
13. OUTPUT
==================================================

Return ONLY a single valid JSON object matching the schema below.

Do NOT include any conversational preamble, intro text, explanations, or prose before or after the JSON.

Do NOT wrap the JSON in markdown code blocks or triple backticks.


Schema:

{{
  "topic": "string",
  "target_level": "beginner | intermediate | advanced | expert",
  "aliases": ["string"],

  "nodes": [
    {{
      "concept_id": "string",

      "level":
        "beginner | intermediate | advanced | expert",

      "node_type":
        "learning | practice",

      "concept_type":
        "code | math | conceptual | other",

      "pattern": "string | null",

      "status":
        "known | partial | learning | locked",

      "action":
        "skip_teach_verify_only | refresher | full_teach",

      "estimated_minutes": 15,

      "target_question_count": 0,

      "depends_on": ["string"]
    }}
  ]
}}


IMPORTANT:

The generated graph must represent a genuine topic-specific
learning structure.

Do not optimize for node count.

Optimize for:

TOPIC COVERAGE
+
LEARNING PROGRESSION
+
PRACTICE COVERAGE
+
PREREQUISITE QUALITY
+
DSA INTERVIEW RELEVANCE.
"""
    ),
    (
        "user",
        """
Topic: {topic}

Target Level: {target_level}

Previous Level Concepts:
{previous_level_concepts}
"""
    )
])

LEARNING_CONTENT_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are an expert DSA instructor creating learning content for an
adaptive learning platform.

Teach the following concept:

Topic: {topic}
Concept: {concept_id}
Level: {level}
Pattern: {pattern}

The learner should understand this concept well enough to solve the
practice problems associated with it.

Create content that includes only what is necessary for this concept.

Include when relevant:

1. Intuition
2. Core idea
3. How the algorithm/data structure works
4. Step-by-step example
5. Code/pseudocode when appropriate
6. Time complexity
7. Space complexity
8. Important edge cases
9. Common mistakes
10. A short checkpoint question

Do not teach downstream concepts.

Do not assume knowledge that is not represented by the prerequisites.

Make the explanation practical and DSA-interview oriented.

Return structured JSON following the provided schema.
{format_instructions}
"""
    ),
    (
        "user",
        """
Topic: {topic}
Concept: {concept_id}
Level: {level}
Pattern: {pattern}

Prerequisites:
{prerequisites}
"""
    )
])

# ---------------------------------------------------------------------------
# 4. QUESTION GENERATION — DSA-specific edge cases & complexity requirements
# ---------------------------------------------------------------------------

QUESTION_GEN_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a DSA question-generation engine for an adaptive learning platform.
You generate assessment questions for a single concept, at a single
difficulty level, optionally tagged to a specific algorithmic pattern.
Return ONLY valid JSON matching the schema below — no prose, no markdown
fences, no explanation.

Schema:
{{
  "questions": [
    {{
      "type": "mcq" | "true_false" | "short_answer" | "open_ended" | "practice",
      "concept_id": string,
      "pattern": string | null,
      "difficulty": integer 1-5,
      "question": string,
      "options": [string] | null,              // required for mcq
      "correct_option": string | null,         // required for mcq/true_false
      "rubric": string | null,                 // required for open_ended/practice
      "expected_time_complexity": string | null,   // e.g. "O(n)" — required for practice + concept_type=code
      "expected_space_complexity": string | null,  // e.g. "O(1)" — required for practice + concept_type=code
      "function_signature": string | null,     // required for practice + concept_type=code
      "test_cases": [{{"input": string, "expected_output": string}}] | null,
      "expected_answer": string | null         // required for short_answer/math
    }}
  ]
}}"""
    ),
    (
        "user",
        """Topic: {topic}
Concept: {concept_id}
Pattern (if any): {pattern}
Concept type: {concept_type}   # code | math | conceptual | other
Target difficulty: {difficulty}   # 1 (easiest) - 5 (hardest), roughly mapped to
                                    # LeetCode Easy (1-2) / Medium (3-4) / Hard (5)
Question type to generate: {question_type}
Number of questions: {count}

Constraints:
- Every question must test ONLY this concept, not prerequisite or future
  concepts.
- Difficulty {difficulty}/5 means: {difficulty_description}
- If concept_type is "code":
    * Must specify an explicit function_signature (name, params, return type).
    * Must specify expected_time_complexity and expected_space_complexity
      the optimal solution should achieve — this is graded separately from
      correctness.
    * test_cases MUST include, at minimum: an empty/trivial input, a
      single-element input, a case with duplicate values, and (if the
      structure supports it) a large-N case suitable for confirming the
      claimed time complexity is actually met rather than a brute-force
      approach that merely produces the right output.
- If concept_type is "math": expected_answer must be a single canonical
  numeric value or canonical simplified symbolic expression.
- If type is "mcq": exactly 4 options, only one correct, distractors must
  represent common DSA misconceptions (e.g. an off-by-one error, confusing
  average-case with worst-case complexity), not obviously wrong answers.
- Do not reuse question phrasing patterns from previous batches for this
  concept: {previous_question_titles}"""
    )
])

# ---------------------------------------------------------------------------
# 5. EVALUATION — complexity is now a first-class graded signal
# ---------------------------------------------------------------------------

EVALUATION_RUBRIC_PROMPT = ChatPromptTemplate.from_template(
    """You are a multi-signal DSA grading evaluator assessing a student's submission.
Concept: {concept_id}
Question: {question}
Expected rubric / solution: {rubric}
Expected time complexity: {expected_time_complexity}
Expected space complexity: {expected_space_complexity}
Student's submission: {answer}
Execution signal feedback (test case pass/fail, if any): {execution_feedback}

Grade against these SEPARATE signals — do not blend them into one vague score:
1. correctness_score — did it pass the test cases / solve the problem?
   (weight this most heavily if execution_feedback is present and reliable)
2. complexity_score — does the student's actual approach achieve the
   expected time/space complexity, or did they pass tests with a
   suboptimal (e.g. brute-force) approach? A correct-but-suboptimal
   solution must NOT score as highly as an optimal one.
3. code_quality_score — readability, naming, structure.
4. edge_case_score — did the submission handle empty/single/duplicate/
   boundary inputs correctly, whether or not the provided test cases
   caught it?

Assign an overall llm_quality_score between 0.0 and 1.0 as a weighted
combination (correctness and complexity should dominate the weighting for
DSA specifically), provide detailed feedback naming which signal(s) were
weak, and return the per-criterion breakdown.

Return structured JSON following format instructions:
{format_instructions}"""
)

# ---------------------------------------------------------------------------
# 6. RE-TEACH — DSA-specific strategy hints
# ---------------------------------------------------------------------------

RETEACH_LESSON_PROMPT = ChatPromptTemplate.from_template(
    """You are an adaptive DSA tutor re-teaching concept '{concept_id}' in topic '{topic}'.
Previous failed explanation strategies for this student: {strategies_tried}
New pedagogical strategy to use: '{strategy}'

Write an engaging, clear lesson explanation using the '{strategy}' technique:
- analogy: relate the mechanism to a concrete real-world process.
- visual_diagram: describe (in text/ASCII) a step-by-step visual trace of
  the data structure/algorithm operating on a small concrete example —
  this is usually the most effective strategy for DSA specifically, since
  the core confusion is almost always about the mechanism over time, not
  the definition.
- interactive_exercise: pose a small trace-through question mid-explanation
  ("pause here — what do you think happens to the pointer next?") before
  revealing the answer.
- worked_example: solve one small concrete problem fully, narrating the
  reasoning at each step, including why the chosen approach achieves its
  time/space complexity.

Explicitly connect the explanation back to time/space complexity — for a
DSA platform, "why is this the complexity" is usually the actual source of
confusion, not "what does this do."

Return structured JSON following format instructions:
{format_instructions}"""
)