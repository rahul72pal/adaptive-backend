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
# 0. DIAGNOSTIC CHECKLIST — fast 1-call concept probe list for diagnostics
# ---------------------------------------------------------------------------

DIAGNOSTIC_CHECKLIST_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a DSA curriculum expert. Generate exactly 15 DSA concept_ids to probe a student's knowledge of "{topic}".

RULES:
1. Return ONLY raw valid JSON — no markdown, no backticks, no prose.
2. Generate exactly 15 concept_ids distributed: 4 beginner + 4 intermediate + 4 advanced + 3 expert.
3. concept_id: snake_case, specific, e.g. array_two_pointer_pattern (not just "arrays" or "basic").
4. Cover the most important/interview-relevant concepts for this DSA topic at each level.
5. Beginner: core mechanics/operations. Intermediate: patterns/techniques. Advanced: optimization/variants. Expert: specialized algorithms.

Output this JSON structure exactly:
{{"topic": "{topic}", "concepts": [{{"concept_id": "...", "level": "beginner|intermediate|advanced|expert"}}]}}"""
    ),
    (
        "user",
        "Generate the 15-concept diagnostic checklist for topic: {topic}"
    )
])

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
6. Standard Markdown Formatting: The "question" string MUST be formatted strictly in clean standard GitHub Flavored Markdown. Use bold (**keyword**) for key requirements, inline code (`variable`, `O(n)`) for parameters and complexity, and fenced code blocks (` ```python ... ``` `) if providing sample inputs or snippets.
   IMPORTANT: Use single quotes `'...'` inside inline code snippets or code blocks (e.g., `s = '((()))'`). NEVER use unescaped double quotes inside string values as they break JSON formatting.

CRITICAL FORMAT REQUIREMENT: Respond ONLY with a valid JSON object matching the format instructions. Do NOT include any intro text, reasoning prose, or conversational commentary outside the JSON. All string values MUST have any internal double quotes escaped or replaced with single quotes.

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
# 2. DIAGNOSTIC — Multi-Agent Evaluation & Feedback Pipeline
# ---------------------------------------------------------------------------

EVALUATOR_AGENT_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a technical DSA Evaluator grading a student's answer for '{concept}'.
Carefully analyze the Question asked and the Student's Answer.

Grading Criteria:
1. Mechanism Correctness: Does the student accurately describe how the data structure or algorithm functions?
2. Complexity Correctness: Does the student correctly identify the Big-O time and/or space complexity required by the question?

Scoring Rubric:
- Known (confidence: 0.80 to 1.00, status: 'known'):
  Both the algorithmic mechanism AND time/space complexity are correct and clearly stated.
  Set mechanism_correct = true, complexity_correct = true, status = 'known'.
- Partial (confidence: 0.40 to 0.75, status: 'partial'):
  The mechanism is mostly correct, but time/space complexity is missing, vague, or inaccurate.
  Set mechanism_correct = true, complexity_correct = false, status = 'partial'.
- Unknown (confidence: 0.00 to 0.35, status: 'unknown'):
  The student's explanation is incorrect, fundamentally confused, or wrong.
  Set mechanism_correct = false, status = 'unknown'.

Be objective and fair: If the student correctly answers the question with the right mechanism and complexity, award confidence >= 0.85 and status 'known'.

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
        """You are an encouraging, world-class DSA Pedagogy & Feedback Specialist Agent.
Your role is to write clear, constructive, and actionable educational feedback for the student based on:
1. The question asked.
2. The student's actual answer.
3. The technical evaluation verdict (confidence, status, mechanism/complexity correctness, gaps).

Pedagogical Structure (formatted strictly in clean standard GitHub Flavored Markdown):
- **Assessment**: Acknowledge the student's answer honestly and encouragingly. If they said "not know" or were unsure, validate their honesty and reassure them that this diagnostic is specifically designed to identify areas for rapid growth.
- **Core Mechanism**: Explain how this concept or operation actually works in 2-3 clear sentences (e.g., how the top pointer operates, why elements don't shift, etc.).
- **Complexity & Operations**: Explicitly state the exact Big-O Time and Space complexities and answer any concrete scenarios in the question (e.g., explain why popping an element from a stack of 1,000 items takes exactly 1 operation and O(1) time).
- **Key Takeaway**: One crisp, memorable bullet point summarizing what to remember for technical interviews.

Writing Guidelines:
- If the student answered "not know", teach them the solution directly so they learn immediately!
- If the student had a misconception (e.g. guessed O(n)), clearly explain why it is constant time or what caused the confusion.
- Never use generic placeholder text like "Response evaluated and confidence updated." Always give concrete DSA facts.

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
    """You are an expert DSA interviewer assessing a student's answer for the concept '{concept}'.
Question asked: {question}
Student's answer: '{answer}'

Grade against these specific signals, not general impression:
1. Correctness of the core mechanism/trace described.
2. Correctness of any time/space complexity claim made or implied — a student who gets the mechanism right but states the wrong Big-O should NOT score as "known."
3. If the student says "not know", "i don't know", "idk", or expresses no knowledge, confidence MUST be 0.0 and status MUST be 'unknown'.

Assign:
- confidence score from 0.0 (complete misconception/no knowledge) to 1.0 (mechanism + complexity both correct and precisely reasoned).
- status ('known' if confidence >= 0.8, 'partial' if 0.4 <= confidence < 0.8, 'unknown' if confidence < 0.4).
- constructive brief feedback written strictly in clean standard Markdown.

CRITICAL FORMAT REQUIREMENT: Respond ONLY with a valid JSON object matching the format instructions.
{format_instructions}"""
)


# ---------------------------------------------------------------------------
# 3. CURRICULUM GRAPH — compact, token-efficient prompt
# ---------------------------------------------------------------------------

CURRICULUM_GRAPH_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a DSA Curriculum Architect. Generate a concept learning graph for a DSA adaptive learning platform.

MANDATORY RULES (all must be followed):
1. OUTPUT: Return ONLY raw valid JSON — no markdown, no backticks, no prose before/after.
2. NODE COUNT: Generate EXACTLY 8 distinct concept nodes (5-6 learning nodes + 2-3 practice nodes) for this level. Every node must be granular and focused on a single specific DSA concept or practice problem.
3. SCOPE: Generate ONLY concepts for level "{target_level}" of topic "{topic}". Do NOT recreate any concept_id from the excluded list.
4. concept_id: Unique, snake_case, descriptive (e.g. array_prefix_sum). No duplicates.
5. depends_on: Reference valid concept_ids from this graph OR excluded (previous) list. No cycles, no self-deps. Root nodes use [].
6. node_type — "learning": teaches a concept/mechanism/complexity. "practice": assesses via problems/implementation. Practice nodes must depend on their prerequisite learning nodes.
7. pattern: Use canonical DSA pattern or "none". Valid tags: two_pointer, sliding_window, fast_slow_pointer, dfs, bfs, backtracking, dp_1d, dp_2d, greedy, binary_search, prefix_sum, divide_conquer, heap, union_find, bit_manipulation, topological_sort, trie. Use a descriptive string for unlisted patterns.
8. concept_type: "code" | "math" | "conceptual" | "mcq" | "short_answer"
9. DIVERSIFICATION FOR PRACTICE NODES: Practice nodes MUST feature a diverse mix of concept_types ("code", "mcq", "conceptual", "short_answer", "math"). Do NOT default all practice nodes to "code". Combine coding challenges with conceptual mechanism analysis and quantitative complexity calculation practice nodes.
10. Defaults for new nodes — status: "learning", action: "full_teach", target_question_count: 0 for learning / 3-5 for practice.
10. QUALITY: Each node must be independently teachable and assessable, DSA-specific, and interview-relevant. No generic fillers (basic_operations, common_patterns, real_world_applications, traversal_mechanisms).
11. COMPLEXITY: Learning nodes must address time/space complexity where meaningful.

LEVEL GUIDANCE for "{target_level}":
- beginner: core terminology, data layout/mechanics, basic operations, simple implementation, O-notation basics
- intermediate: canonical patterns, standard LeetCode-style problem families, combining foundational concepts
- advanced: optimization, edge-case reasoning, advanced variants, non-obvious problem transformations
- expert: specialized algorithms, multi-pattern combinations, novel problem design, trade-off analysis

{format_instructions}"""
    ),
    (
        "user",
        """Topic: {topic}
Level: {target_level}
Excluded concept_ids (DO NOT generate any of these): {previous_level_concepts}

Output the concept graph JSON for "{topic}" at level "{target_level}" now."""
    )
])

LEARNING_CONTENT_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a Principal DSA Instructor building interactive lesson content.

Your task is to teach: '{concept_id}' under topic '{topic}' (Level: {level}, Pattern: {pattern}).

Pedagogical Directives:

1. Target Scope: Cover strictly what is required to master '{concept_id}'.
   Keep explanations crisp, clear, and direct. "Intuition" and "Core
   Principle" must NOT restate each other:
     - Intuition = one short, informal "why this exists" framing (an
       analogy or real-world parallel), no technical mechanism detail.
     - Core Principle = the precise, formal technical definition/mechanism.
   If you find yourself writing near-identical sentences in both, merge
   the content into Core Principle and make Intuition purely the analogy.

2. Language-Specific Implementation: Provide code in {language}
   (python, cpp, js) that is GENUINELY COMPILABLE/RUNNABLE AS WRITTEN --
   this is a hard requirement, not a style preference:
     - Include every required operator, statement terminator, and header/
       import. Never truncate or drop syntax elements (e.g. `cout << x`
       must include the `<<`, not just `cout x`).
     - The snippet must be a complete, self-contained unit (a function or
       class plus a minimal call/usage line) that would run without
       modification if pasted into a compiler/interpreter.
     - Include inline comments highlighting the core mechanism only --
       do not comment every line.

3. Explanation Language: Write intuitions, core ideas, mechanisms,
   step-by-step examples, common mistakes, and checkpoint questions in
   {explanation_language} (english or hinglish).
   - If 'hinglish': natural Hinglish (Hindi in Roman script mixed with
     technical English terms), e.g. "Array ek contiguous memory layout hai
     jisme elements index se access hote hain..."
   - If 'english': clear technical English.

4. Concrete Step-by-Step Walkthrough: Trace a small, real data example
   across 3-4 sequential states, showing the actual state change at each
   step (not just narrating that a change happens).

5. Complexity Reporting: Report `time_complexity` as a single value UNLESS
   this concept genuinely has different best-case vs worst-case behavior
   (e.g. a search that can exit early) -- only in that case, also set
   `has_case_variance: true` and fill `best_case`/`worst_case` separately.
   For concepts with no such variance (e.g. a full traversal that always
   visits every element), set `has_case_variance: false` and leave
   `best_case`/`worst_case` null -- do not fill them with a duplicate value.

6. Measurable Checkpoint -- must test a REAL distinguishing trade-off or
   boundary case, not a restated definition:
     - BAD example (do not do this): "Do we need to visit all nodes?"
       with options that just restate yes/no -- this tests nothing.
     - GOOD example: "If you forget to update the pointer inside the loop
       in a linked list traversal, what happens?" with options covering
       infinite loop / early termination / compile error / no effect --
       this tests understanding of the actual failure mode.
   Provide exactly 4 options as structured {{"id": ..., "text": ...}}
   objects using ids "A", "B", "C", "D". Do NOT prefix or embed a letter/
   label inside the `text` field itself (e.g. write "Infinite loop
   occurs", NOT "A. Infinite loop occurs") -- the rendering layer applies
   the label separately, and embedding it yourself causes duplicated/
   garbled labels downstream. Set `correct_option_id` to just the id
   ("A", "B", "C", or "D").

7. Standard Markdown Content Formatting: Format all text fields (`intuition`, `core_idea`, `mechanism`, `step_by_step_example`, `common_mistakes`) strictly using standard GitHub Flavored Markdown (bold key terms `**...**`, inline code `code`, markdown ordered/unordered lists, and clean linebreaks).

Output Format:
Return ONLY valid JSON matching the schema. No markdown fences, no
conversational prose.
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
        """You are an elite Computer Science educator and principal software engineer creating a comprehensive, beautifully structured concept lesson.
Your task is to teach '{concept_id}' under topic '{topic}' (Level: {level}, Pattern: {pattern}).

Pedagogical Directives:
1. Target Scope: Cover strictly what is required to master '{concept_id}'.
   - Intuition: One short, intuitive analogy or real-world mental model.
   - Core Principle: Formal technical definition and operational mechanism.
   - Step-by-Step Walkthrough: Trace a small, real data example step-by-step showing state transitions.
   - Implementation: Provide fully working, runnable code in {language} with imports, comments, and clear logic.
   - Complexity Analysis: Exact Big-O Time Complexity (best/worst) and Space Complexity with clear mechanical reasons.
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
[Clean, runnable code]
```

## ⏱️ Complexity Analysis
- **Time Complexity (Best Case):** `O(...)`
- **Time Complexity (Worst Case):** `O(...)`
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
   - Intuition = one short, informal "why this exists" framing (analogy or real-world parallel), no code details here.
   - Core Idea = the precise, formal technical definition/mechanism in 2-3 sentences.
   - Mechanism = step-by-step operational logic (1., 2., 3.).

2. Language-Specific Implementation: Provide code in {language} (python, cpp, js) that is GENUINELY COMPILABLE/RUNNABLE as written:
   - Include every required operator, statement terminator, and header/import.
   - Never truncate syntax.

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
4. Measurable Checkpoint: 4 options testing a real failure mode or tradeoff. Set correct_option to matching text.

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

Language & Style Directives:
1. Programming Language: {language} (python | cpp | js).
   - If concept_type is "code" or function_signature is required, generate code/signature in {language}.
   - For 'cpp': Use C++ syntax, e.g. `int solution(vector<int>& nums)`.
   - For 'js': Use JavaScript syntax, e.g. `function solution(nums)`.
   - For 'python': Use Python syntax, e.g. `def solution(nums: list[int]) -> int:`.
2. Text & Explanation Language: {explanation_language} (english | hinglish).
   - If explanation_language is 'hinglish', write question prompts, options, explanations, and descriptions in Hinglish (Hindi written in Roman script with English technical terms).
   - If explanation_language is 'english', write in standard technical English.
3. Standard Markdown Formatting: The `question` text MUST be formatted strictly in clean standard GitHub Flavored Markdown (use bold `**...**` for key requirements, inline code `O(n)` for complexity, clear markdown paragraphs, and fenced code blocks for sample data).
4. CRITICAL MCQ FORMAT RULES:
   - For `type: "mcq"` or `"true_false"`, the `question` field MUST contain ONLY the question prompt/problem statement. DO NOT append or include options (like "A) ...", "B) ...") in the `question` field!
   - The `options` list MUST contain the FULL textual choices (e.g. `["Constant O(1) time", "Linear O(n) time", "Logarithmic O(log n) time", "Quadratic O(n²) time"]`). NEVER return single letters `["A", "B", "C", "D"]` in `options`!
   - The `correct_option` field MUST match the exact text of the correct choice from the `options` array (e.g. `"Constant O(1) time"`).

Schema:
{{
  "questions": [
    {{
      "type": "mcq" | "true_false" | "short_answer" | "open_ended" | "practice",
      "concept_id": string,
      "pattern": string | null,
      "difficulty": integer 1-5,
      "question": string,                      // ONLY the question text; NEVER include options here
      "options": [string] | null,              // Full descriptive text for each option; required for mcq/true_false
      "correct_option": string | null,         // Exact text of the correct option matching an item in options
      "rubric": string | null,                 // required for open_ended/practice
      "expected_time_complexity": string | null,   // e.g. "O(n)" — required for practice + concept_type=code
      "expected_space_complexity": string | null,  // e.g. "O(1)" — required for practice + concept_type=code
      "function_signature": string | null,     // required for practice + concept_type=code
      "test_cases": [{{"input": string, "expected_output": string}}] | null,
      "expected_answer": string | null         // required for short_answer/math
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
      CRITICAL RULE FOR "input": Provide ONLY valid raw JSON array or value without variable assignment!
      Good: "[]", "[1]", "[2, 2, 1, 1, 1, 2, 2]"
      BAD: "nums = []", "nums = [1]*1000", "range(10)"
    * "expected_output": The exact return value as string, e.g. "-1", "1", "2"."""
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