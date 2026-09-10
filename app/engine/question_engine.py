import uuid
import random
import logging
import asyncio
from typing import Dict, Any, List, Optional
from app.llm.gateway import call_llm_gateway
from app.llm.prompts import QUESTION_GEN_PROMPT
from app.llm.schemas import QuestionBatchSchema

logger = logging.getLogger(__name__)

def sanitize_question_for_client(q: Dict[str, Any]) -> Dict[str, Any]:
    """Strips grading rubrics and correct answers before sending to the client."""
    safe_q = dict(q)
    safe_q.pop("correct_option", None)
    safe_q.pop("rubric", None)
    safe_q.pop("expected_answer", None)
    return safe_q

async def get_or_generate_question(
    topic: str,
    concept_id: str,
    concept_type: str = "conceptual",
    difficulty: int = 3,
    mongo_db: Any = None
) -> Dict[str, Any]:
    """Retrieves a single client-safe question for topic & concept_id from MongoDB or triggers mini-batch LLM generation."""
    batch = await get_or_generate_question_batch(
        topic=topic,
        concept_id=concept_id,
        concept_type=concept_type,
        difficulty=difficulty,
        count=3,
        mongo_db=mongo_db
    )
    if batch:
        return random.choice(batch)
    
    fallback_id = f"q_fallback_{uuid.uuid4().hex[:6]}"
    fallback_doc = {
        "_id": fallback_id,
        "id": fallback_id,
        "topic": topic,
        "concept_id": concept_id,
        "concept_type": concept_type,
        "difficulty": difficulty,
        "type": "mcq",
        "question": f"What is a core principle of {concept_id} in {topic}?",
        "options": ["Correct principle implementation", "Invalid approach A", "Invalid approach B", "Invalid approach C"],
        "execution_method": "llm_rubric"
    }
    return sanitize_question_for_client(fallback_doc)

async def get_or_generate_question_batch(
    topic: str,
    concept_id: str,
    concept_type: str = "conceptual",
    difficulty: int = 3,
    count: int = 3,  # Target 3-5 for adaptive steps
    mongo_db: Any = None
) -> List[Dict[str, Any]]:
    """Retrieves or creates a pool of questions, returning client-safe versions."""
    q_collection = mongo_db.get_collection("questions") if mongo_db is not None else None
    topic_clean = topic.strip().lower()

    existing_questions = []

    # 1. Query existing questions pool matching topic, concept & difficulty
    if q_collection is not None:
        filter_query = {
            "concept_id": concept_id,
            "topic": topic_clean,
            "difficulty": difficulty
        }
        try:
            existing_questions = await asyncio.wait_for(
                q_collection.find(filter_query).to_list(20),
                timeout=3.0
            )
        except Exception as e:
            logger.warning(f"Failed to read questions from MongoDB: {e}")
            existing_questions = []

    # 2. Return cached pool if enough questions exist
    if len(existing_questions) >= count:
        random.shuffle(existing_questions)
        return [sanitize_question_for_client(q) for q in existing_questions[:count]]

    # 3. Cache miss/low pool -> generate a small mini-batch of questions (3-5)
    needed = count - len(existing_questions)
    previous_titles = [q.get("question", "")[:50] for q in existing_questions]

    try:
        res = await call_llm_gateway(
            QUESTION_GEN_PROMPT,
            {
                "topic": topic_clean,
                "concept_id": concept_id,
                "concept_type": concept_type,
                "difficulty": difficulty,
                "question_type": "practice" if concept_type == "code" else "mcq",
                "count": min(needed, 3),  # Keeps tokens within safe limits
                "previous_question_titles": previous_titles
            },
            QuestionBatchSchema
        )

        new_docs = []
        for q_item in getattr(res, "questions", []):
            q_type = getattr(q_item, "type", "mcq") or "mcq"
            
            if concept_type == "code" and q_type == "practice":
                method = "piston_api"
            elif concept_type == "math":
                method = "sympy" if q_type == "practice" else "numeric_tolerance"
            elif q_type in ["mcq", "true_false"]:
                method = "exact_match"
            else:
                method = "llm_rubric"

            q_id = f"q_{uuid.uuid4().hex[:8]}"
            doc = {
                "_id": q_id,
                "id": q_id,
                "topic": topic_clean,
                "concept_id": concept_id,
                "concept_type": concept_type,
                "difficulty": getattr(q_item, "difficulty", difficulty) or difficulty,
                "type": q_type,
                "question": q_item.question,
                "options": getattr(q_item, "options", None),
                "correct_option": getattr(q_item, "correct_option", None),
                "rubric": getattr(q_item, "rubric", None),
                "test_cases": [
                    tc.model_dump() if hasattr(tc, "model_dump") else tc 
                    for tc in q_item.test_cases
                ] if getattr(q_item, "test_cases", None) else None,
                "expected_answer": getattr(q_item, "expected_answer", None),
                "execution_method": getattr(q_item, "execution_method", None) or method
            }
            new_docs.append(doc)

        if new_docs and q_collection is not None:
            try:
                await asyncio.wait_for(q_collection.insert_many(new_docs), timeout=3.0)
                existing_questions.extend(new_docs)
            except Exception as insert_err:
                logger.warning(f"Could not persist questions: {insert_err}")
                existing_questions.extend(new_docs)
        elif new_docs:
            existing_questions.extend(new_docs)

    except Exception as err:
        logger.error(f"Batch generation failed for '{concept_id}': {err}", exc_info=True)

    # 4. Fallback filler if pool is still below requested count
    while len(existing_questions) < count:
        idx = len(existing_questions) + 1
        fallback_id = f"q_fallback_{concept_id}_{idx}_{uuid.uuid4().hex[:4]}"
        fallback_doc = {
            "_id": fallback_id,
            "id": fallback_id,
            "topic": topic_clean,
            "concept_id": concept_id,
            "concept_type": concept_type,
            "difficulty": difficulty,
            "type": "mcq",
            "question": f"Which statement best applies to '{concept_id}' in {topic}?",
            "options": [
                f"Standard established implementation of {concept_id}",
                f"Deprecated pattern in {concept_id}",
                "Invalid execution behavior",
                "Unrelated syntax component"
            ],
            "correct_option": f"Standard established implementation of {concept_id}",
            "rubric": "Evaluates fundamental understanding.",
            "test_cases": None,
            "expected_answer": None,
            "execution_method": "exact_match"
        }
        if q_collection is not None:
            try:
                await q_collection.insert_one(fallback_doc)
            except Exception as fe:
                logger.warning(f"Could not insert fallback doc: {fe}")
        existing_questions.append(fallback_doc)

    return [sanitize_question_for_client(q) for q in existing_questions[:count]]

