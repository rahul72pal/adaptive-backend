import uuid
import random
import logging
import asyncio
from typing import Dict, Any, List, Optional
from app.llm.gateway import call_llm_gateway
from app.llm.prompts import (
    QUESTION_GEN_PROMPT,
    QUESTION_META_PROMPT,
    QUESTION_STATEMENT_PROMPT,
    TEST_CASE_GEN_PROMPT,
)
from app.llm.schemas import QuestionBatchSchema, QuestionTextSchema, TestCaseListSchema

import re

logger = logging.getLogger(__name__)

DIFFICULTY_DESCRIPTIONS = {
    1: "definition-recall level, testing basic core concepts and syntax",
    2: "simple direct application of concept with straightforward examples",
    3: "intermediate problem-solving and standard usage scenarios",
    4: "advanced concept application involving edge cases and complex scenarios",
    5: "expert mastery requiring combining this concept with unusual edge cases or complex scenarios"
}

def normalize_mcq_question_fields(q_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Cleans up questions where options (e.g. A) ..., B) ...) are crammed inside
    the question text and options list contains only letters like ['C', 'D', 'A', 'B'].
    Separates the stem question into 'question' and assigns the real text choices into 'options'.
    """
    if not isinstance(q_dict, dict):
        return q_dict

    q_type = q_dict.get("type", "mcq")
    if q_type not in ("mcq", "true_false"):
        return q_dict

    raw_q_text = str(q_dict.get("question") or "").strip()
    raw_opts = list(q_dict.get("options") or [])
    raw_corr = str(q_dict.get("correct_option") or "").strip()

    extracted = []
    clean_lines = []
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
        q_dict["question"] = "\n".join(clean_lines).strip()
        q_dict["options"] = [text for _, text in extracted]
        resolved_corr = raw_corr
        for key, text in extracted:
            if raw_corr.upper() == key or raw_corr == text or raw_corr.upper().startswith(key):
                resolved_corr = text
                break
        q_dict["correct_option"] = resolved_corr

    return q_dict

def build_question_markdown(q: Dict[str, Any]) -> str:
    """Formats question details into clean GitHub Flavored Markdown."""
    if not isinstance(q, dict):
        return str(q or "")

    cid = q.get("concept_id", "Practice Problem").replace("_", " ").title()
    prompt = q.get("question", "") or q.get("prompt", "") or q.get("title", "")
    diff = q.get("difficulty", 3)
    exp_time = q.get("expected_time_complexity")
    exp_space = q.get("expected_space_complexity")
    test_cases = q.get("test_cases") or []
    options = q.get("options") or []

    lines = [f"### 🎯 {cid} (Difficulty Level {diff})\n"]
    lines.append(f"{prompt}\n")

    if options and len(options) > 0:
        lines.append("#### Options:\n")
        for i, opt in enumerate(options):
            letter = chr(65 + i)
            opt_text = opt.get("text") if isinstance(opt, dict) else str(opt)
            lines.append(f"- **{letter}.** {opt_text}")
        lines.append("")

    if exp_time or exp_space:
        lines.append("#### ⏱️ Complexity Constraints")
        if exp_time:
            lines.append(f"- **Target Time Complexity:** `{exp_time}`")
        if exp_space:
            lines.append(f"- **Target Auxiliary Space:** `{exp_space}`")
        lines.append("")

    if test_cases and len(test_cases) > 0:
        lines.append("#### 🧪 Sample Test Cases")
        for i, tc in enumerate(test_cases[:3], 1):
            inp = tc.get("input", "")
            out = tc.get("expected_output", "")
            lines.append(f"- **Case {i}:** Input: `{inp}` → Expected Output: `{out}`")
        lines.append("")

    return "\n".join(lines).strip()

def sanitize_question_for_client(q: Dict[str, Any]) -> Dict[str, Any]:
    """Strips grading rubrics, correct answers, and internal Mongo _id before sending to the client."""
    safe_q = dict(q)
    safe_q = normalize_mcq_question_fields(safe_q)
    safe_q.pop("correct_option", None)
    safe_q.pop("rubric", None)
    safe_q.pop("expected_answer", None)
    safe_q.pop("hidden_test_cases", None)
    if safe_q.get("_id") and not safe_q.get("id"):
        safe_q["id"] = str(safe_q["_id"])
    safe_q.pop("_id", None)
    safe_q["markdown"] = build_question_markdown(safe_q)
    return safe_q


def question_kind_for(concept_type: str) -> str:
    """Map a concept type onto the question kind used by the fast pipeline."""
    ct = (concept_type or "").lower()
    if ct in ("code", "practice"):
        return "practice"
    if ct in ("math", "conceptual", "mcq"):
        return "mcq"
    return "mcq"


def _is_code_question(concept_type: str, question_type: str) -> bool:
    return question_type == "practice" or concept_type in ("code", "practice")


async def generate_question_statement(
    topic: str,
    concept_id: str,
    concept_type: str = "conceptual",
    difficulty: int = 3,
    language: str = "python",
    explanation_language: str = "english",
) -> str:
    """Stream-friendly plain-text statement generation (no JSON)."""
    kind = question_kind_for(concept_type)
    chunks: List[str] = []
    from app.llm.gateway import call_llm_gateway_stream

    async for delta in call_llm_gateway_stream(
        QUESTION_STATEMENT_PROMPT,
        {
            "topic": topic,
            "concept_id": concept_id,
            "concept_type": concept_type,
            "difficulty": difficulty,
            "question_type": kind,
            "language": language,
            "explanation_language": explanation_language,
        }
    ):
        chunks.append(delta)
    return "".join(chunks).strip()


async def generate_question_meta(
    topic: str,
    concept_id: str,
    concept_type: str = "conceptual",
    difficulty: int = 3,
    language: str = "python",
    statement: Optional[str] = None,
) -> QuestionTextSchema:
    """Small structured call: type/options/signature/rubric only."""
    kind = question_kind_for(concept_type)
    if not statement:
        statement = await generate_question_statement(
            topic, concept_id, concept_type, difficulty, language
        )
    return await call_llm_gateway(
        QUESTION_META_PROMPT,
        {
            "topic": topic,
            "concept_id": concept_id,
            "concept_type": concept_type,
            "difficulty": difficulty,
            "question_type": kind,
            "language": language,
            "statement": statement,
        },
        QuestionTextSchema,
    )


def _clean_test_cases(raw_cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cleaned: List[Dict[str, Any]] = []
    seen = set()
    for tc in raw_cases or []:
        if not isinstance(tc, dict):
            continue
        inp = str(tc.get("input", "")).strip()
        out = str(tc.get("expected_output", "")).strip()
        if not out:
            continue
        key = (inp, out)
        if key in seen:
            continue
        seen.add(key)
        cleaned.append({"input": inp, "expected_output": out})
    return cleaned[:5]


async def generate_test_cases(
    topic: str,
    concept_id: str,
    statement: str,
    function_signature: Optional[str],
    language: str = "python",
    input_adapter: str = "json_args_v1",
    test_case_schema: str = "args_kwargs_json_v1",
    max_attempts: int = 2,
) -> List[Dict[str, Any]]:
    """Generate validated test cases with retry; never raises."""
    for attempt in range(1, max_attempts + 1):
        try:
            res = await call_llm_gateway(
                TEST_CASE_GEN_PROMPT,
                {
                    "statement": statement,
                    "function_signature": function_signature or "(infer from the statement)",
                    "language": language,
                    "input_adapter": input_adapter,
                    "test_case_schema": test_case_schema,
                },
                TestCaseListSchema,
            )
            cases = _clean_test_cases([
                tc.model_dump() if hasattr(tc, "model_dump") else dict(tc)
                for tc in getattr(res, "test_cases", [])
            ])
            if cases:
                logger.info(
                    "[question_engine] Generated %d test cases for '%s' (attempt %d)",
                    len(cases), concept_id, attempt,
                )
                return cases
        except Exception as exc:
            logger.warning(
                "[question_engine] Test-case generation attempt %d failed for '%s': %s",
                attempt, concept_id, exc,
            )
    logger.error("[question_engine] No test cases generated for '%s'.", concept_id)
    return []


def build_question_doc(
    meta: QuestionTextSchema,
    statement: str,
    topic_clean: str,
    concept_id: str,
    concept_type: str,
    difficulty: int,
    test_cases: Optional[List[Dict[str, Any]]] = None,
    language: str = "python",
) -> Dict[str, Any]:
    """Assemble a persisted/internal question document from pipeline parts."""
    q_type = getattr(meta, "type", None) or question_kind_for(concept_type)
    is_code = _is_code_question(concept_type, q_type)

    options = list(getattr(meta, "options", None) or [])
    correct = getattr(meta, "correct_option", None)
    if options:
        if correct is not None and correct not in options and str(correct).upper() in ["A", "B", "C", "D"]:
            idx = ["A", "B", "C", "D"].index(str(correct).upper())
            if idx < len(options):
                correct = options[idx]
        if correct not in options:
            correct = options[0]
        # Keep the correct option content stable while shuffling position.
        correct_text = correct
        random.shuffle(options)
        correct = correct_text

    q_id = f"q_{uuid.uuid4().hex[:10]}"
    doc: Dict[str, Any] = {
        "_id": q_id,
        "id": q_id,
        "topic": topic_clean,
        "concept_id": concept_id,
        "concept_type": concept_type,
        "difficulty": int(getattr(meta, "difficulty", difficulty) or difficulty),
        "type": q_type,
        "question": statement or getattr(meta, "question", ""),
        "options": options or None,
        "correct_option": correct,
        "rubric": getattr(meta, "rubric", None),
        "expected_time_complexity": getattr(meta, "expected_time_complexity", None),
        "expected_space_complexity": getattr(meta, "expected_space_complexity", None),
        "function_signature": getattr(meta, "function_signature", None),
        "execution_mode": getattr(meta, "execution_mode", None),
        "input_adapter": getattr(meta, "input_adapter", None),
        "output_adapter": getattr(meta, "output_adapter", None),
        "supported_languages": getattr(meta, "supported_languages", None),
        "test_case_schema": getattr(meta, "test_case_schema", None),
        "comparison_mode": getattr(meta, "comparison_mode", None),
        "problem_version": getattr(meta, "problem_version", None),
        "absolute_tolerance": getattr(meta, "absolute_tolerance", None),
        "relative_tolerance": getattr(meta, "relative_tolerance", None),
        "starter_code": getattr(meta, "starter_code", None),
        "expected_answer": getattr(meta, "expected_answer", None),
        "explanation": getattr(meta, "explanation", None),
        # Keep a small public subset for the UI; the rest are judged only by
        # the server when a submission is finalized.
        "test_cases": (test_cases or [])[:2] or None,
        "hidden_test_cases": (test_cases or [])[2:] or None,
        "language": language,
        "execution_method": "onecompiler" if is_code else "exact_match",
        "created_at": __import__("datetime").datetime.utcnow().isoformat(),
    }
    return normalize_mcq_question_fields(doc)


async def _find_cached_question(
    concept_id: str,
    topic_clean: str,
    difficulty: int,
    concept_type: str,
    q_collection: Any,
    language: Optional[str] = None,
    exclude_question_ids: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    if q_collection is None:
        return None
    try:
        docs = await asyncio.wait_for(
            q_collection.find({"concept_id": concept_id, "topic": topic_clean}).to_list(12),
            timeout=2.0,
        )
    except Exception as exc:
        logger.warning("[question_engine] Cache read failed: %s", exc)
        return None

    excluded_ids = {str(question_id) for question_id in (exclude_question_ids or [])}
    docs = [
        d for d in docs
        if not str(d.get("_id", "")).startswith("q_fallback_")
        and not str(d.get("id", "")).startswith("q_fallback_")
        and str(d.get("id") or d.get("_id") or "") not in excluded_ids
        and (not language or not d.get("language") or str(d.get("language")).lower() == language.lower())
        and (not d.get("concept_type") or str(d.get("concept_type")).lower() == concept_type.lower())
    ]
    if not docs:
        return None

    is_code = concept_type in ("code", "practice")
    if is_code:
        with_tests = [d for d in docs if d.get("test_cases")]
        # Cached code questions without tests are useless for execution.
        docs = with_tests
    if not docs:
        return None

    exact = [d for d in docs if d.get("difficulty") == difficulty]
    return random.choice(exact or docs)


async def generate_single_question(
    topic: str,
    concept_id: str,
    concept_type: str = "conceptual",
    difficulty: int = 3,
    language: str = "python",
    explanation_language: str = "english",
    statement: Optional[str] = None,
    mongo_db: Any = None,
) -> Dict[str, Any]:
    """Generate ONE question via the split pipeline and persist it."""
    topic_clean = (topic or "").strip().lower()
    language = (language or "python").lower()
    is_code = concept_type in ("code", "practice")

    if not statement:
        statement = await generate_question_statement(
            topic_clean, concept_id, concept_type, difficulty, language, explanation_language
        )

    # Metadata is required; fall back to a minimal deterministic meta on failure.
    try:
        meta = await generate_question_meta(
            topic_clean, concept_id, concept_type, difficulty, language, statement
        )
    except Exception as exc:
        logger.warning("[question_engine] Meta generation failed for '%s': %s", concept_id, exc)
        meta = QuestionTextSchema(
            type="practice" if is_code else "mcq",
            concept_id=concept_id,
            difficulty=difficulty,
            question=statement or f"Practice {concept_id}",
            options=None if is_code else ["Option A", "Option B", "Option C", "Option D"],
            correct_option=None if is_code else "Option A",
            rubric=f"Evaluate correctness for {concept_id}." if is_code else None,
        )

    test_cases: List[Dict[str, Any]] = []
    if is_code:
        test_cases = await generate_test_cases(
            topic_clean,
            concept_id,
            statement or meta.question,
            getattr(meta, "function_signature", None),
            language,
            getattr(meta, "input_adapter", "json_args_v1"),
            getattr(meta, "test_case_schema", "args_kwargs_json_v1"),
        )

    doc = build_question_doc(
        meta=meta,
        statement=statement or meta.question,
        topic_clean=topic_clean,
        concept_id=concept_id,
        concept_type=concept_type,
        difficulty=difficulty,
        test_cases=test_cases,
        language=language,
    )

    q_collection = mongo_db.get_collection("questions") if mongo_db is not None else None
    if q_collection is not None:
        try:
            await asyncio.wait_for(q_collection.insert_one(dict(doc)), timeout=4.0)
        except Exception as exc:
            logger.warning("[question_engine] Could not persist question: %s", exc)

    return doc


async def get_or_generate_question(
    topic: str,
    concept_id: str,
    concept_type: str = "conceptual",
    difficulty: int = 3,
    language: str = "python",
    explanation_language: str = "english",
    mongo_db: Any = None,
    include_internal: bool = False
) -> Dict[str, Any]:
    """Retrieve a cached question or generate exactly one (fast path)."""
    topic_clean = (topic or "").strip().lower()
    q_collection = mongo_db.get_collection("questions") if mongo_db is not None else None

    cached = await _find_cached_question(
        concept_id, topic_clean, difficulty, concept_type, q_collection, language
    )
    if cached:
        doc = dict(cached)
    else:
        doc = await generate_single_question(
            topic=topic_clean,
            concept_id=concept_id,
            concept_type=concept_type,
            difficulty=difficulty,
            language=language,
            explanation_language=explanation_language,
            mongo_db=mongo_db,
        )

    doc = normalize_mcq_question_fields(doc)
    return doc if include_internal else sanitize_question_for_client(doc)

async def _generate_and_save_batch(
    topic_clean: str,
    concept_id: str,
    concept_type: str,
    difficulty: int,
    needed: int,
    previous_titles: list,
    q_collection: Any,
    language: str = "python",
    explanation_language: str = "english"
) -> List[Dict[str, Any]]:
    from app.catalog.context import TopicGenerationContext
    context = TopicGenerationContext.from_topic_identifier(
        topic_clean,
        selected_language=language
    )
    diff_desc = DIFFICULTY_DESCRIPTIONS.get(difficulty, "standard conceptual understanding")
    
    # Adapt concept_type and question_type to topic assessment mode
    if context.assessment_mode == "conceptual_only" and concept_type == "code":
        eff_concept_type = "conceptual"
    else:
        eff_concept_type = concept_type

    q_type_to_gen = "practice" if eff_concept_type == "code" else "mcq"
    eff_lang = context.selected_language or language or "python"

    res = await call_llm_gateway(
        QUESTION_GEN_PROMPT,
        {
            "topic": context.canonical_topic,
            "concept_id": concept_id,
            "pattern": context.pattern_tags[0] if context.pattern_tags else "none",
            "concept_type": eff_concept_type,
            "difficulty": difficulty,
            "difficulty_description": diff_desc,
            "question_type": q_type_to_gen,
            "count": min(needed, 3),  # Keeps tokens within safe limits
            "previous_question_titles": str(previous_titles) if previous_titles else "None",
            "language": eff_lang,
            "explanation_language": explanation_language
        },
        QuestionBatchSchema
    )

    new_docs = []
    for q_item in getattr(res, "questions", []):
        q_type = getattr(q_item, "type", "mcq") or "mcq"
        
        if eff_concept_type == "code" and q_type == "practice":
            method = "onecompiler"
        elif eff_concept_type == "math":
            method = "sympy" if q_type == "practice" else "numeric_tolerance"
        elif q_type in ["mcq", "true_false"]:
            method = "exact_match"
        else:
            method = "llm_rubric"

        raw_opts = getattr(q_item, "options", None)
        raw_correct = getattr(q_item, "correct_option", None)
        if raw_opts and isinstance(raw_opts, list) and len(raw_opts) > 1 and raw_correct:
            raw_opts = list(raw_opts)
            c_str = str(raw_correct).strip()
            if c_str.upper() in ["A", "B", "C", "D"] and len(raw_opts) == 4:
                raw_correct = raw_opts[["A", "B", "C", "D"].index(c_str.upper())]
            random.shuffle(raw_opts)

        q_id = f"q_{uuid.uuid4().hex[:8]}"
        generated_cases = [
            tc.model_dump() if hasattr(tc, "model_dump") else tc
            for tc in (getattr(q_item, "test_cases", None) or [])
        ]
        doc = {
            "_id": q_id,
            "id": q_id,
            "topic_id": context.topic_id,
            "topic": topic_clean,
            "canonical_topic": context.canonical_topic,
            "concept_id": concept_id,
            "concept_type": eff_concept_type,
            "difficulty": getattr(q_item, "difficulty", difficulty) or difficulty,
            "type": q_type,
            "question": q_item.question,
            "options": raw_opts,
            "correct_option": raw_correct,
            "rubric": getattr(q_item, "rubric", None),
            "test_cases": generated_cases[:2] if generated_cases else None,
            "hidden_test_cases": generated_cases[2:] if q_type == "practice" and generated_cases else None,
            "function_signature": getattr(q_item, "function_signature", None),
            "execution_mode": "function" if q_type == "practice" else None,
            "input_adapter": "legacy_json_v1" if q_type == "practice" else None,
            "output_adapter": "legacy_python_v1" if q_type == "practice" else None,
            "supported_languages": [language] if q_type == "practice" else None,
            "test_case_schema": "legacy_json_v1" if q_type == "practice" else None,
            "comparison_mode": "legacy_json_v1" if q_type == "practice" else None,
            "problem_version": 1 if q_type == "practice" else None,
            "expected_answer": getattr(q_item, "expected_answer", None),
            "execution_method": getattr(q_item, "execution_method", None) or method,
            "metadata_version": context.metadata_version
        }
        doc = normalize_mcq_question_fields(doc)
        new_docs.append(doc)

    if new_docs and q_collection is not None:
        try:
            await asyncio.wait_for(q_collection.insert_many(new_docs), timeout=5.0)
        except Exception as insert_err:
            logger.warning(f"Could not persist questions: {insert_err}")

    return new_docs

async def get_or_generate_question_batch(
    topic: str,
    concept_id: str,
    concept_type: str = "conceptual",
    difficulty: int = 3,
    count: int = 3,  # Target 3-5 for adaptive steps
    language: str = "python",
    explanation_language: str = "english",
    mongo_db: Any = None,
    include_internal: bool = False
) -> List[Dict[str, Any]]:
    """Retrieves or creates a pool of questions, returning client-safe or internal versions."""
    from app.catalog.context import TopicGenerationContext
    context = TopicGenerationContext.from_topic_identifier(topic, selected_language=language)
    q_collection = mongo_db.get_collection("questions") if mongo_db is not None else None
    topic_clean = topic.strip().lower()

    def _prepare_q(q_item: Dict[str, Any]) -> Dict[str, Any]:
        normalized = normalize_mcq_question_fields(dict(q_item))
        if include_internal:
            return normalized
        return sanitize_question_for_client(normalized)

    existing_questions = []

    # 1. Query existing questions pool matching topic, concept & difficulty
    if q_collection is not None:
        filter_query = {
            "concept_id": concept_id,
            "$or": [
                {"topic": topic_clean},
                {"topic_id": context.topic_id},
                {"canonical_topic": context.canonical_topic}
            ],
            "difficulty": difficulty
        }
        try:
            raw_existing = await asyncio.wait_for(
                q_collection.find(filter_query).to_list(20),
                timeout=3.0
            )
            existing_questions = [
                q for q in raw_existing
                if not str(q.get("_id", "")).startswith("q_fallback_")
                and not str(q.get("id", "")).startswith("q_fallback_")
            ]
            if not existing_questions:
                # Fallback: query without strict difficulty constraint
                raw_existing = await asyncio.wait_for(
                    q_collection.find({"concept_id": concept_id, "topic": topic_clean}).to_list(20),
                    timeout=3.0
                )
                existing_questions = [
                    q for q in raw_existing
                    if not str(q.get("_id", "")).startswith("q_fallback_")
                    and not str(q.get("id", "")).startswith("q_fallback_")
                ]
        except Exception as e:
            logger.warning(f"Failed to read questions from MongoDB: {e}")
            existing_questions = []

    # 2. Return cached pool if enough questions exist
    if len(existing_questions) >= count:
        random.shuffle(existing_questions)
        return [_prepare_q(q) for q in existing_questions[:count]]

    # If we have at least 1 question, return what we have instantly and replenish in background!
    if existing_questions:
        needed = count - len(existing_questions)
        previous_titles = [q.get("question", "")[:50] for q in existing_questions]
        asyncio.create_task(
            _generate_and_save_batch(
                topic_clean, concept_id, concept_type, difficulty, needed, previous_titles, q_collection, language, explanation_language
            )
        )
        random.shuffle(existing_questions)
        return [_prepare_q(q) for q in existing_questions]

    # 3. Cache miss (0 questions) -> generate a small mini-batch synchronously
    needed = count
    previous_titles = []

    try:
        new_docs = await _generate_and_save_batch(
            topic_clean, concept_id, concept_type, difficulty, needed, previous_titles, q_collection, language, explanation_language
        )
        existing_questions.extend(new_docs)
    except Exception as err:
        logger.error(f"Batch generation failed for '{concept_id}': {err}", exc_info=True)
        # Create a safe fallback doc if LLM call failed
        q_id = f"q_fallback_{uuid.uuid4().hex[:8]}"
        fallback_doc = {
            "_id": q_id,
            "id": q_id,
            "topic": topic_clean,
            "concept_id": concept_id,
            "concept_type": concept_type,
            "difficulty": difficulty,
            "type": "mcq",
            "question": f"Which strategy is recommended for mastering {concept_id.replace('_', ' ')}?",
            "options": [
                "Apply standard algorithm with edge case validation",
                "Ignore memory limits and recompute endlessly",
                "Hardcode fixed static values",
                "Bypass data structures completely"
            ],
            "correct_option": "Apply standard algorithm with edge case validation",
            "execution_method": "exact_match"
        }
        existing_questions.append(fallback_doc)

    if not existing_questions:
        q_id = f"q_fallback_{uuid.uuid4().hex[:8]}"
        fallback_doc = {
            "_id": q_id,
            "id": q_id,
            "topic": topic_clean,
            "concept_id": concept_id,
            "concept_type": concept_type,
            "difficulty": difficulty,
            "type": "mcq",
            "question": f"Which strategy is recommended for mastering {concept_id.replace('_', ' ')}?",
            "options": [
                "Apply standard algorithm with edge case validation",
                "Ignore memory limits and recompute endlessly",
                "Hardcode fixed static values",
                "Bypass data structures completely"
            ],
            "correct_option": "Apply standard algorithm with edge case validation",
            "execution_method": "exact_match"
        }
        existing_questions.append(fallback_doc)

    return [_prepare_q(q) for q in existing_questions[:count]]


