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
# 0. DIAGNOSTIC CHECKLIST — fast concept probe list for diagnostics
# ---------------------------------------------------------------------------

DIAGNOSTIC_CHECKLIST_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an expert curriculum architect. Generate a list of concept_ids to probe a student's knowledge of "{topic}".

TOPIC CONTEXT:
Topic: {topic}
Kind: {topic_kind}
Assessment Mode: {assessment_mode}
Target Probe Count: {max_probes}

RULES:
1. Return ONLY raw valid JSON — no markdown, no backticks, no prose before or after.
2. Generate EXACTLY {max_probes} concept_ids distributed across proficiency levels:
   - beginner: core terminology, data/concept structure, basic mechanics
   - intermediate: canonical patterns, typical problem variations, standard techniques
   - advanced: edge-case optimizations, advanced variants, trade-off analysis
   - expert: specialized algorithms, novel extensions, or complex edge scenarios
3. concept_id: snake_case, specific, descriptive (e.g. binary_search_lower_bound, not just "search").
4. Cover the most important/interview-relevant concepts for this specific topic kind.
5. If the topic is conceptual/math, focus on theoretical foundations, invariants, and mathematical mechanics. If code-based, focus on algorithmic mechanisms and complexity.

Output this JSON structure exactly:
{{"topic": "{topic}", "concepts": [{{"concept_id": "...", "level": "beginner|intermediate|advanced|expert"}}]}}"""
    ),
    (
        "user",
        "Generate the {max_probes}-concept diagnostic checklist for topic: {topic}"
    )
])

# ---------------------------------------------------------------------------
# 1. DIAGNOSTIC — ask node
# ---------------------------------------------------------------------------

DIAGNOSTIC_ASK_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an expert technical interviewer probing a student's real understanding of the concept: '{concept}', within the topic: '{topic}'.

TOPIC CONTEXT & ASSESSMENT DIRECTIVES:
- Topic Kind: {topic_kind}
- Assessment Mode: {assessment_mode}
- Target Language (if code): {selected_language}

Your task is to generate ONE precise, highly targeted diagnostic question.

Question Guidelines:
1. Test Deep Understanding, Not Rote Definition:
   Never ask "What is {concept}?" or textbook definitions. Probe the underlying mechanism, invariant, mathematical reasoning, or operational state transitions.
2. Assessment Mode Directives:
   - If assessment_mode is 'code' or 'mixed': Probe the mechanical state trace, memory behavior, or time/space complexity. Frame the question around a small concrete example.
   - If assessment_mode is 'conceptual_only' or topic is theoretical: Probe logical correctness, boundary conditions, correctness invariant, or mathematical reasoning. Do NOT demand irrelevant code execution or Big-O if not applicable to this concept.
3. Concise & Direct Answerability:
   The question must be answerable in 2-4 precise sentences so capability can be accurately evaluated without the student needing to write an entire code file.
4. Strictly Scoped:
   Stay strictly focused on '{concept}' — do not require knowledge of unrelated concepts.
5. Standard Markdown Formatting:
   The "question" string MUST be formatted strictly in clean standard GitHub Flavored Markdown. Use bold (**keyword**) for key requirements, inline code (`variable`, `O(n)`) for parameters, and single quotes `'...'` inside code snippets. NEVER use unescaped double quotes inside string values.

CRITICAL FORMAT REQUIREMENT: Respond ONLY with a valid JSON object matching the format instructions.
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
# 2. DIAGNOSTIC — Multi-Agent Evaluation & Feedback Pipeline
# ---------------------------------------------------------------------------

EVALUATOR_AGENT_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a technical assessment evaluator objectively grading a student's answer for '{concept}'.
Carefully analyze the Question asked and the Student's Answer.

Assessment Context:
- Topic Kind: {topic_kind}
- Assessment Mode: {assessment_mode}

Grading Criteria:
1. Mechanism / Conceptual Correctness: Does the student accurately describe how the concept/algorithm operates or correctly state the core invariant/reasoning?
2. Complexity / Quantitative Precision: If Big-O time/space complexity or mathematical bound was asked or expected for this concept, is it stated accurately? (If the question or topic is conceptual/math and did not require Big-O, mark complexity_correct as true).

Scoring Rubric (Enforce continuous, explicit boundaries with NO gaps):
- Known (confidence: 0.80 to 1.00, status: 'known'):
  The mechanism/conceptual foundation is correct and clearly stated. If complexity was applicable, it is also correct.
  Set mechanism_correct = true, complexity_correct = true, status = 'known'.
- Partial (confidence: 0.40 to 0.79, status: 'partial'):
  The core mechanism is partially understood or mostly correct, but has minor misconceptions, missing edge cases, or inaccurate complexity.
  Set mechanism_correct = true, complexity_correct = false, status = 'partial'.
- Unknown (confidence: 0.00 to 0.39, status: 'unknown'):
  The student's explanation is incorrect, fundamentally confused, or expresses no knowledge.
  Set mechanism_correct = false, status = 'unknown'.

Be objective and fair. Award confidence matching the true depth of understanding.

CRITICAL FORMAT REQUIREMENT: Respond ONLY with a valid JSON object matching format instructions.
{format_instructions}"""
    ),
    (
        "user",
        """Target Concept: {concept}
Question Asked: {question}
Student's Answer: {answer}"""
    )
])

FEEDBACK_AGENT_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a concise, encouraging educational Feedback Specialist.
Your task is to write SHORT, SIMPLE, and DIRECT feedback (strictly 2 to 3 sentences maximum, under 50 words).

Guidelines:
1. Verdict: In 1 short sentence, state whether the answer was correct, partially correct, or missing key elements.
2. Core Takeaway: In 1-2 concise sentences, state the correct mechanism, core invariant, or complexity for '{concept}'.
3. Tone: Encouraging, supportive, and punchy.
4. DO NOT write long essays or verbose section headers. Keep it clean and easy to read at a glance!

CRITICAL FORMAT REQUIREMENT: Respond ONLY with a valid JSON object matching the format instructions.
{format_instructions}"""
    ),
    (
        "user",
        """Target Concept: {concept}
Question Asked: {question}
Student's Answer: {answer}
Evaluation Grade:
- Confidence: {confidence}
- Status: {status}
- Mechanism Correct: {mechanism_correct}
- Complexity Correct: {complexity_correct}
- Identified Gaps: {key_gaps}
- Verdict Summary: {evaluation_summary}"""
    )
])

DIAGNOSTIC_SCORE_PROMPT = ChatPromptTemplate.from_template(
    """You are an expert interviewer assessing a student's answer for the concept '{concept}'.
Question asked: {question}
Student's answer: '{answer}'

Grade against these objective criteria:
1. Correctness of the core mechanism, invariant, or conceptual reasoning described.
2. Correctness of complexity claims where applicable.
3. If the student indicates no knowledge ("not know", "idk", blank), confidence MUST be 0.0 and status MUST be 'unknown'.

Rubric Boundaries:
- 'known': confidence in [0.80, 1.00]
- 'partial': confidence in [0.40, 0.79]
- 'unknown': confidence in [0.00, 0.39]

Assign:
- confidence score from 0.0 to 1.0.
- status ('known', 'partial', 'unknown').
- constructive brief feedback in clean standard Markdown.

CRITICAL FORMAT REQUIREMENT: Respond ONLY with a valid JSON object matching format instructions.
{format_instructions}"""
)


# ---------------------------------------------------------------------------
# 3. CURRICULUM GRAPH — compact, token-efficient prompt
# ---------------------------------------------------------------------------

CURRICULUM_GRAPH_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a Curriculum Architect. Generate a concept learning graph for an adaptive learning platform.

MANDATORY RULES:
1. OUTPUT: Return ONLY raw valid JSON — no markdown, no backticks, no prose before/after.
2. NODE COUNT: Generate {target_node_count} distinct concept nodes ({learning_count} learning nodes + {practice_count} practice nodes) for this level. Every node must be granular and focused on a single specific concept or practice milestone.
3. SCOPE & TAXONOMY:
   - Topic: "{topic}" | Kind: {topic_kind} | Granularity Scope: {topic_scope}
   - Assessment Mode: {assessment_mode}
   - Generate ONLY concepts for level "{target_level}".
   - Do NOT recreate any concept_id from the excluded list.
   - For broad topics, decompose into representative foundational pillars.
   - For leaf algorithms (e.g. Kadane's algorithm), focus on algorithm variants, invariants, and edge cases without unnecessary bloat.
   - For concrete interview problems, preserve problem constraints, modeling transformations, and optimizations.
   - For theoretical or systems-practice topics, focus on conceptual models and operational characteristics.
4. concept_id: Unique, snake_case, descriptive (e.g. array_prefix_sum). No duplicates.
5. depends_on: Reference valid concept_ids from this graph OR excluded (previous) list. No cycles, no self-deps. Root nodes use [].
6. node_type — "learning": teaches a concept/mechanism/complexity. "practice": assesses via problems/implementation. Practice nodes must depend on their prerequisite learning nodes.
7. pattern: Use canonical pattern tag or "none" (e.g. two_pointer, sliding_window, dfs, bfs, dp_1d, greedy, binary_search, or topic-appropriate tag).
8. concept_type: "code" | "math" | "conceptual" | "mcq" | "short_answer"
9. PRACTICE NODES SPECIFICATION:
   - If assessment_mode is "conceptual_only": Practice nodes MUST feature conceptual, mcq, short_answer, or math concept_types. Do NOT force code execution.
   - If assessment_mode is "code" or "mixed": Combine coding challenges with conceptual mechanism analysis and quantitative complexity calculation.
10. Defaults for new nodes — status: "learning", action: "full_teach", target_question_count: 0 for learning / 3-5 for practice.
11. QUALITY: Each node must be independently teachable and assessable. Avoid generic fillers (e.g. basic_operations, common_patterns).
12. COMPLEXITY: Learning nodes must address time/space complexity or mathematical bounds where meaningful.

LEVEL GUIDANCE for "{target_level}":
- beginner: core terminology, data/concept mechanics, basic operations, simple implementation, foundational bounds
- intermediate: canonical patterns, typical problem variations, standard techniques, combining foundational concepts
- advanced: optimization, edge-case reasoning, advanced variants, trade-off analysis
- expert: specialized extensions, multi-paradigm combinations, novel problem design

{format_instructions}"""
    ),
    (
        "user",
        """Topic: {topic}
Kind: {topic_kind}
Level: {target_level}
Target Nodes: {target_node_count} ({learning_count} learning, {practice_count} practice)
Assessment Mode: {assessment_mode}
Excluded concept_ids (DO NOT generate any of these): {previous_level_concepts}

Output the concept graph JSON for "{topic}" at level "{target_level}" now."""
    )
])

LEARNING_CONTENT_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a Principal Technical Instructor building interactive lesson content.

Your task is to teach: '{concept_id}' under topic '{topic}' (Level: {level}, Pattern: {pattern}).

TOPIC CONTEXT:
- Topic Kind: {topic_kind}
- Assessment Mode: {assessment_mode}

Pedagogical Directives:

1. Target Scope: Cover strictly what is required to master '{concept_id}'.
   Keep explanations crisp, clear, and direct. "Intuition" and "Core Principle" must NOT restate each other:
     - Intuition = one short, informal "why this exists" framing (an analogy or real-world parallel), no code details here.
     - Core Principle = the precise, formal technical definition, invariant, or operational mechanism.

2. Language-Specific Implementation / Technical Representation:
   - If assessment_mode is 'code' or 'mixed': Provide code in {language} (python, cpp, js) that is GENUINELY COMPILABLE/RUNNABLE AS WRITTEN:
     * Include every required operator, statement terminator, and header/import.
     * The snippet must be a complete, self-contained unit that would run if pasted into an interpreter/compiler.
     * Include inline comments highlighting the core mechanism only.
   - If assessment_mode is 'conceptual_only': Provide a clear pseudo-code representation, mathematical definition, or theoretical illustration. Do not force irrelevant compiler syntax.

3. Explanation Language: Write intuitions, core ideas, mechanisms, step-by-step examples, common mistakes, and checkpoint questions in {explanation_language} (english or hinglish).

4. Concrete Step-by-Step Walkthrough: Trace a small, real data example or state transition across 3-4 sequential steps showing the actual state change at each step.

5. Complexity Reporting: Report `time_complexity` and `space_complexity` where meaningful, or "N/A" if purely theoretical.

6. Measurable Checkpoint: Must test a REAL distinguishing trade-off, invariant, or boundary case.
   Provide exactly 4 options as structured {{"id": ..., "text": ...}} objects using ids "A", "B", "C", "D". Do NOT embed letter labels inside the text field. Set `correct_option_id` to just the id ("A", "B", "C", or "D").

7. Standard Markdown Content Formatting: Format all text fields strictly using standard GitHub Flavored Markdown.

Output Format:
Return ONLY valid JSON matching the schema. No markdown fences, no conversational prose.
{format_instructions}"""
    ),
    (
        "user",
        """Topic: {topic}
Concept: {concept_id}
Level: {level}
Pattern: {pattern}
Prerequisites: {prerequisites}
Coding Language: {language}
Explanation Language: {explanation_language}"""
    )
])


LESSON_MARKDOWN_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an elite Computer Science educator creating a comprehensive, beautifully structured concept lesson.
Your task is to teach '{concept_id}' under topic '{topic}' (Level: {level}, Pattern: {pattern}).

Pedagogical Directives:
1. Target Scope: Cover strictly what is required to master '{concept_id}'.
   - Intuition: One short, intuitive analogy or mental model.
   - Core Principle: Formal technical definition and operational mechanism or invariant.
   - Step-by-Step Walkthrough: Trace a small, real data example step-by-step showing state transitions.
   - Implementation / Code: Provide clean, working code in {language} (or pseudo-code/formal notation if conceptual).
   - Complexity Analysis: Exact Time and Space Complexity with clear mechanical reasons (or theoretical bounds).
   - Edge Cases & Common Pitfalls: Key boundary conditions and common mistakes.

2. Explanation Language: Write all explanations in {explanation_language} (english or hinglish).

3. Output Format:
Output clean, standard GitHub Flavored Markdown directly.
Do NOT output JSON. Do NOT wrap the entire response in outer markdown code fences.
Structure your lesson cleanly using:
# {concept_id}

> 💡 **Intuition:** [Analogy or mental model]

## 🎯 Core Principle
[Precise technical mechanism and definition]

## ⚙️ Operational Mechanism & Walkthrough
[Step-by-step trace of a concrete example]

## 💻 Implementation ({language})
```{language}
[Clean code or pseudo-code]
```

## ⏱️ Complexity Analysis
- **Time Complexity:** `O(...)`
- **Auxiliary Space:** `O(...)`
- **Mechanical Reason:** [Explanation]

## ⚠️ Key Edge Cases
- [Key edge case 1]
- [Key edge case 2]

## 🚫 Common Pitfalls
- [Common mistake 1]
- [Common mistake 2]
"""
    ),
    (
        "user",
        """Topic: {topic}
Concept: {concept_id}
Level: {level}
Pattern: {pattern}
Prerequisites: {prerequisites}
Coding Language: {language}
Explanation Language: {explanation_language}"""
    )
])



LESSON_OVERVIEW_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an elite Computer Science educator creating the core overview for a concept.
Your task is to teach the foundational overview of '{concept_id}' under topic '{topic}' (Level: {level}, Pattern: {pattern}).

Pedagogical Directives:
1. Target Scope: Cover strictly what is required to master '{concept_id}'.
   - Intuition = one short, informal "why this exists" framing (analogy or real-world parallel).
   - Core Idea = the precise, formal technical definition/mechanism in 2-3 sentences.
   - Mechanism = step-by-step operational logic (1., 2., 3.).

2. Implementation: Provide code in {language} (python, cpp, js) if code/mixed, or clean pseudo-logic if conceptual.

3. Explanation Language: Write all explanations in {explanation_language} (english or hinglish).

Output Format:
Return ONLY valid JSON matching the schema. No markdown fences.
{format_instructions}"""
    ),
    (
        "user",
        """Topic: {topic}
Concept: {concept_id}
Level: {level}
Pattern: {pattern}
Prerequisites: {prerequisites}
Coding Language: {language}
Explanation Language: {explanation_language}"""
    )
])


LESSON_DEEP_DIVE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an elite Computer Science educator creating the deep-dive walkthrough, complexity analysis, and checkpoint question.
Your task is to provide the deep dive for '{concept_id}' under topic '{topic}' (Level: {level}, Pattern: {pattern}).

Pedagogical Directives:
1. Step-by-Step Example: Trace a small real data example across 3-4 sequential states showing actual value transitions.
2. Complexity Analysis: Provide best_case, worst_case, space, and a mechanical reason.
3. Edge Cases & Common Mistakes: Name 2 real boundary conditions and 2 common student pitfalls.
4. Measurable Checkpoint: 4 options testing a real failure mode, invariant, or tradeoff. Set correct_option to matching text.

Explanation Language: {explanation_language} (english or hinglish).

Output Format:
Return ONLY valid JSON matching the schema. No markdown fences.
{format_instructions}"""
    ),
    (
        "user",
        """Topic: {topic}
Concept: {concept_id}
Level: {level}
Pattern: {pattern}
Prerequisites: {prerequisites}
Coding Language: {language}
Explanation Language: {explanation_language}"""
    )
])


# ---------------------------------------------------------------------------
# 4. QUESTION GENERATION — Metadata-tailored assessment engine
# ---------------------------------------------------------------------------

QUESTION_GEN_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a question-generation engine for an adaptive learning platform.
You generate assessment questions for a single concept, at a single difficulty level.
Return ONLY valid JSON matching the schema below — no prose, no markdown fences.

Language & Style Directives:
1. Programming Language: {language} (python | cpp | js).
   - If concept_type is "code" or function_signature is required, generate signature in {language}.
   - If concept_type is "conceptual" or "math", focus on reasoning, trace, invariant, or MCQ.
2. Text & Explanation Language: {explanation_language} (english | hinglish).
3. Standard Markdown Formatting: The `question` text MUST be formatted strictly in clean standard GitHub Flavored Markdown.
4. CRITICAL MCQ FORMAT RULES:
   - For `type: "mcq"` or `"true_false"`, the `question` field MUST contain ONLY the question prompt. DO NOT include options in `question`!
   - The `options` list MUST contain full textual choices. NEVER return single letters in `options`!
   - The `correct_option` field MUST match the exact text of the correct choice from the `options` array.

Schema:
{{
  "questions": [
    {{
      "type": "mcq" | "true_false" | "short_answer" | "open_ended" | "practice",
      "concept_id": string,
      "pattern": string | null,
      "difficulty": integer 1-5,
      "question": string,
      "options": [string] | null,
      "correct_option": string | null,
      "rubric": string | null,
      "expected_time_complexity": string | null,
      "expected_space_complexity": string | null,
      "function_signature": string | null,
      "test_cases": [{{"input": string, "expected_output": string}}] | null,
      "expected_answer": string | null
    }}
  ]
}}

{format_instructions}"""
    ),
    (
        "user",
        """Topic: {topic}
Concept: {concept_id}
Pattern (if any): {pattern}
Concept type: {concept_type}   # code | math | conceptual | other
Target difficulty: {difficulty}   # 1 (easiest) - 5 (hardest)
Question type to generate: {question_type}
Number of questions: {count}
Coding Language: {language}
Explanation Language: {explanation_language}

Constraints:
- If concept_type is "code" and type is "practice":
    * "function_signature": MUST be provided in {language} with clear types.
    * "execution_method": Set strictly to "browser_compiler".
    * "test_cases": MUST provide 3 to 4 cases.
      Provide ONLY valid raw JSON array or value without variable assignment!
    * "expected_output": The exact return value as string."""
    )
])

# ---------------------------------------------------------------------------
# 5. EVALUATION — Complexity & correctness multi-signal evaluator
# ---------------------------------------------------------------------------

EVALUATION_RUBRIC_PROMPT = ChatPromptTemplate.from_template(
    """You are a multi-signal grading evaluator assessing a student's submission.
Concept: {concept_id}
Question: {question}
Expected rubric / solution: {rubric}
Expected time complexity: {expected_time_complexity}
Expected space complexity: {expected_space_complexity}
Student's submission: {answer}
Execution signal feedback (test case pass/fail, if any): {execution_feedback}

Grade against these SEPARATE signals:
1. correctness_score — did it pass the test cases or solve the problem conceptually?
2. complexity_score — does the approach achieve expected bounds? If question is purely conceptual, base on correctness of reasoning.
3. code_quality_score — clarity, naming, logical structure.
4. edge_case_score — did submission handle boundary conditions?

Assign an overall llm_quality_score between 0.0 and 1.0 as a weighted combination, provide detailed feedback naming which signal(s) were weak, and return the per-criterion breakdown.

Return structured JSON following format instructions:
{format_instructions}"""
)

# ---------------------------------------------------------------------------
# 6. RE-TEACH — Adaptive Remediation & Strategy Hints
# ---------------------------------------------------------------------------

RETEACH_LESSON_PROMPT = ChatPromptTemplate.from_template(
    """You are an adaptive tutor re-teaching concept '{concept_id}' in topic '{topic}'.
Previous failed explanation strategies for this student: {strategies_tried}
New pedagogical strategy to use: '{strategy}'

Write an engaging, clear lesson explanation using the '{strategy}' technique:
- analogy: relate the mechanism to a concrete real-world process.
- visual_diagram: describe (in text/ASCII) a step-by-step visual trace of the concept or state transitions.
- interactive_exercise: pose a small trace-through checkpoint mid-explanation before revealing the answer.
- worked_example: solve one small concrete problem fully, narrating the reasoning at each step.

Explicitly target the demonstrated student misconceptions without simply repeating previous verbatim text.

Return structured JSON following format instructions:
{format_instructions}"""
)


# ---------------------------------------------------------------------------
# 7. FAST QUESTION PIPELINE — statement → meta/code → test cases
# ---------------------------------------------------------------------------
# The pipeline is split so the UI can stream the problem statement immediately
# while the smaller metadata call and the test-case tool run afterwards.

QUESTION_STATEMENT_PROMPT = ChatPromptTemplate.from_template(
    """You write a single assessment question for an adaptive learning platform.

Topic: {topic}
Concept: {concept_id}
Concept type: {concept_type}   # code | math | conceptual | other
Target difficulty: {difficulty} (1=easiest, 5=hardest)
Question kind: {question_type}
Programming language: {language}
Explanation language: {explanation_language}

Write ONLY the problem statement in clean GitHub-flavoured Markdown.
Rules:
- For a coding practice question: describe the task, input/output contract, and
  1-2 examples. Do NOT include the function signature, starter code, or test cases.
- For a multiple-choice question: write ONLY the question stem. Do NOT write the
  answer choices.
- Keep it concise: at most 12 lines. Use `inline code` for identifiers.
- No preamble, no headings like "Question:", no JSON, no code fences around the whole answer.

Return the statement text only."""
)

QUESTION_META_PROMPT = ChatPromptTemplate.from_template(
    """You convert a question statement into structured metadata.
Topic: {topic}
Concept: {concept_id}
Concept type: {concept_type}
Difficulty: {difficulty}
Question kind: {question_type}
Programming language: {language}
Statement:
{statement}

Produce JSON with these fields:
- type: "practice" for coding tasks, otherwise "mcq" | "true_false" | "short_answer" | "open_ended"
- concept_id: "{concept_id}"
- difficulty: {difficulty}
- question: the statement verbatim (Markdown)
- options: EXACTLY 4 full-text choices when type is mcq/true_false, else null
- correct_option: exact text of the correct option (must match one of options) for mcq, else null
- rubric: grading rubric for practice/open_ended, else null
- expected_time_complexity / expected_space_complexity: e.g. "O(n)" for coding tasks, else null
- function_signature: the function name and parameters the tests will call, in {language}
  (e.g. "def two_sum(nums: list[int], target: int) -> list[int]:"), else null
- For coding tasks declare execution_mode="function", input_adapter="json_args_v1"
  for ordinary JSON arguments, output_adapter="json_value_v1", supported_languages=["{language}"],
  test_case_schema="args_kwargs_json_v1", comparison_mode="exact_json_v1", problem_version=2.
  Use "python_linked_list_args_v1" / "node_value_v1" for linked-list tasks,
  "python_tree_level_order_args_v1" for binary trees, or
  "python_graph_adjacency_args_v1" for graph adjacency tasks only when language is Python.
  For JavaScript use json_args_v1/json_value_v1. Never infer from input shape.
- starter_code: for practice tasks ONLY, complete self-contained starter code in
  {language} that already includes everything the learner needs, else null.
  * Put the function name on the LAST line of the file so the runner can find it.
  * Include all required imports (e.g. `from typing import List, Optional`).
  * If the task involves a linked list or binary tree, DEFINE the node class(es)
    in the starter code exactly as:
      class ListNode:
          def __init__(self, val=0, next=None):
              self.val = val
              self.next = next
      class TreeNode:
          def __init__(self, val=0, left=None, right=None):
              self.val = val
              self.left = left
              self.right = right
    Test inputs for those types are JSON level-order/value arrays; do not ask the
    learner to parse them.
  * End the function body with a placeholder (`pass`, `return None`, or a TODO).
  * Do NOT include test cases or driver code.
- expected_answer: short reference answer for short_answer, else null
- explanation: one-sentence explanation of the correct choice, else null

{format_instructions}"""
)

TEST_CASE_GEN_PROMPT = ChatPromptTemplate.from_template(
    """Generate test cases for a coding problem.

Problem statement:
{statement}

Function signature ({language}):
{function_signature}
Declared input adapter: {input_adapter}
Declared test-case schema: {test_case_schema}

Rules:
- Return 3 to 5 test cases.
- `input` MUST be JSON object with `args` array and `kwargs` object, e.g.
  {{"args":[[2,7,11,15],9],"kwargs":{{}}}}. Never emit positional argument guesses.
- For linked lists, encode a typed argument as {{"$type":"linked_list","values":[...],"cycle_index":null}};
  use an empty values array for an empty list and -1 or null for no cycle.
- For binary trees, encode {{"$type":"binary_tree_level_order","serialization":"level_order_nulls_v1","values":[...] }}.
- For graph adapters, encode {{"$type":"graph_adjacency","node_count":N,"edges":[[u,v]],"directed":false}}.
- `expected_output` MUST be the exact JSON value the function returns
  (e.g. "[0, 1]", "true", "42", "\"abc\"").
- Include at least one edge case (empty input, duplicates, negatives, or a single element).
- Double-check each expected output by mentally executing a correct solution.

{format_instructions}"""
)
