import logging
import uuid
from typing import Optional, Any, List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel

from app.db.postgres import get_db
from app.db.mongo import get_mongo_db
from app.core.dependencies import get_current_user
from app.db.models import User, DiagnosticSession, ConceptMastery
from app.engine.curriculum_engine import (
    generate_personalized_learning_path,
    generate_next_level_graph,
    generate_learning_content,
    sync_learning_path_progress,
    record_user_completed_node,
    slugify_topic,
    get_available_nodes
)
from app.engine.mastery_engine import check_level_completion
from app.graphs.reteach_graph import select_strategy_node

logger = logging.getLogger(__name__)

from app.llm.stream_events import (
    MessageStartEvent,
    TextDeltaEvent,
    UserInputEvent,
    UserInputPayload,
    InputOption,
    ValidationConstraints,
    MessageEndEvent,
    ErrorEvent,
    ToolStartEvent,
    ToolEndEvent,
    format_sse,
    validate_user_input_submission,
)

router = APIRouter(prefix="/api/learning", tags=["Learning Path & Curriculum Engine"])

class GeneratePathSchema(BaseModel):
    topic: str
    target_level: str = "expert"  # beginner, intermediate, advanced, expert
    diagnostic_session_id: Optional[str] = None
    language: Optional[str] = "python"
    explanation_language: Optional[str] = "english"

class CompleteLevelSchema(BaseModel):
    learning_path_id: str

class CompleteNodeSchema(BaseModel):
    topic: str
    concept_id: str
    learning_path_id: Optional[str] = None
    path_id: Optional[str] = None
    graph_id: Optional[str] = None

class LessonRequestSchema(BaseModel):
    topic: str
    concept_id: str
    level: str = "beginner"
    pattern: Optional[str] = "none"
    prerequisites: List[str] = []
    language: Optional[str] = "python"
    explanation_language: Optional[str] = "english"
    force_regenerate: bool = False

class CheckpointAnswerSchema(BaseModel):
    topic: str
    concept_id: str
    input_id: str
    answer: str

class ReteachRequestSchema(BaseModel):
    topic: str
    concept_id: str
    strategies_tried: List[str] = []
    consecutive_failures: int = 1

@router.post("/generate-path", status_code=status.HTTP_201_CREATED)
async def generate_path(
    payload: GeneratePathSchema,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    mongo_db: Any = Depends(get_mongo_db)
):
    """Generates personalized learning path starting with the beginner level graph."""
    logger.info(
        f"[generate_path] Request topic='{payload.topic}', target_level='{payload.target_level}', "
        f"diag_session='{payload.diagnostic_session_id}' from user_id={current_user.id}"
    )

    knowledge_profile = {}
    diag_uuid = None

    if payload.diagnostic_session_id:
        try:
            diag_uuid = uuid.UUID(payload.diagnostic_session_id)
            stmt = select(DiagnosticSession).where(DiagnosticSession.id == diag_uuid)
            res = await db.execute(stmt)
            session_rec = res.scalar_one_or_none()
            if session_rec:
                knowledge_profile = session_rec.knowledge_profile or {}
        except ValueError as err:
            logger.warning(f"[generate_path] Invalid diagnostic_session_id format: {err}")

    try:
        path_doc = await generate_personalized_learning_path(
            user_id=current_user.id,
            topic=payload.topic,
            target_level=payload.target_level,
            knowledge_profile=knowledge_profile,
            diagnostic_session_id=diag_uuid,
            mongo_db=mongo_db,
            db=db
        )
        return path_doc
    except Exception as e:
        logger.error(f"[generate_path] Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate learning path: {str(e)}")

@router.post("/complete-level")
async def complete_level_and_progress(
    payload: CompleteLevelSchema,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    mongo_db: Any = Depends(get_mongo_db)
):
    """Triggers generation of the next level graph when the user completes the current level."""
    try:
        updated_doc = await generate_next_level_graph(
            learning_path_id=payload.learning_path_id,
            user_id=current_user.id,
            mongo_db=mongo_db,
            db=db
        )
        return updated_doc
    except ValueError as val_err:
        raise HTTPException(status_code=404, detail=str(val_err))
    except Exception as e:
        logger.error(f"[complete_level_and_progress] Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to progress level: {str(e)}")

@router.post("/complete-node")
async def complete_node(
    payload: CompleteNodeSchema,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    mongo_db: Any = Depends(get_mongo_db)
):
    """Marks a concept node as mastered, updates Postgres ConceptMastery, and syncs learning path progress in MongoDB."""
    topic_slug = slugify_topic(payload.topic)
    concept_id = payload.concept_id.strip().lower()

    try:
        # 1. Update concept mastery in Postgres
        stmt = select(ConceptMastery).where(
            ConceptMastery.user_id == current_user.id,
            ConceptMastery.topic == topic_slug,
            ConceptMastery.concept_id == concept_id
        )
        res = await db.execute(stmt)
        mastery_record = res.scalar_one_or_none()

        if not mastery_record:
            mastery_record = ConceptMastery(
                user_id=current_user.id,
                topic=topic_slug,
                concept_id=concept_id,
                mastery=1.0,
                attempts=1,
                failures=0,
                status="mastered"
            )
            db.add(mastery_record)
        else:
            mastery_record.status = "mastered"
            mastery_record.mastery = max(mastery_record.mastery, 1.0)
            mastery_record.attempts = max(mastery_record.attempts, 1)

        # 2. Record completed node in user_topics table/collection for this user
        await record_user_completed_node(
            user_id=current_user.id,
            topic_slug=topic_slug,
            concept_id=concept_id,
            mongo_db=mongo_db,
            db=db
        )

        # 3. Sync learning path progress (unlocks downstream nodes, updates current_node)
        lp_id = payload.learning_path_id or payload.path_id or payload.graph_id
        updated_doc = await sync_learning_path_progress(
            user_id=current_user.id,
            topic_slug=topic_slug,
            mongo_db=mongo_db,
            db=db,
            learning_path_id=lp_id
        )

        if not updated_doc:
            raise HTTPException(status_code=404, detail="Learning path not found for given user and topic.")

        return updated_doc
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[complete_node] Error completing node '{concept_id}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to complete node: {str(e)}")

from fastapi.responses import StreamingResponse
import json

def build_lesson_markdown(content: dict) -> str:
    """Formats structured lesson content into clean, comprehensive GitHub Flavored Markdown."""
    if not isinstance(content, dict):
        return str(content or "")

    cid_raw = content.get("concept_id") or "Core Concept"
    cid_title = cid_raw.replace("_", " ").title()
    lang = (content.get("language") or "python").lower()

    lines = [f"# {cid_title}\n"]

    intuition = content.get("intuition")
    if intuition and str(intuition).strip():
        lines.append(f"> 💡 **Intuition:** {str(intuition).strip()}\n")

    core_idea = content.get("core_idea")
    if core_idea and str(core_idea).strip():
        lines.append(f"## 🎯 Core Principle\n{str(core_idea).strip()}\n")

    mechanism = content.get("mechanism")
    if mechanism and str(mechanism).strip():
        lines.append(f"## ⚙️ Operational Mechanism\n{str(mechanism).strip()}\n")

    example = content.get("step_by_step_example")
    if example and str(example).strip():
        lines.append(f"## 🔍 Step-by-Step Walkthrough\n{str(example).strip()}\n")

    code = content.get("code_snippet")
    if code and str(code).strip():
        lines.append(f"## 💻 Implementation ({lang.upper()})\n```{lang}\n{str(code).strip()}\n```\n")

    complexity = content.get("complexity")
    if isinstance(complexity, dict):
        best = complexity.get("best_case", "O(...)")
        worst = complexity.get("worst_case", "O(...)")
        space = complexity.get("space", "O(...)")
        expl = complexity.get("explanation", "")
        lines.append("## ⏱️ Complexity Analysis\n")
        lines.append(f"- **Time Complexity (Best Case):** `{best}`")
        lines.append(f"- **Time Complexity (Worst Case):** `{worst}`")
        lines.append(f"- **Auxiliary Space:** `{space}`")
        if expl:
            lines.append(f"- **Mechanical Reason:** {expl}")
        lines.append("")

    edge_cases = content.get("edge_cases")
    if isinstance(edge_cases, list) and len(edge_cases) > 0:
        lines.append("## ⚠️ Key Edge Cases\n")
        for ec in edge_cases:
            lines.append(f"- {ec}")
        lines.append("")

    common_mistakes = content.get("common_mistakes")
    if isinstance(common_mistakes, list) and len(common_mistakes) > 0:
        lines.append("## 🚫 Common Pitfalls\n")
        for cm in common_mistakes:
            lines.append(f"- {cm}")
        lines.append("")

    checkpoint = content.get("checkpoint")
    if isinstance(checkpoint, dict) and checkpoint.get("question"):
        lines.append("## ❓ Checkpoint Question\n")
        lines.append(f"**{checkpoint.get('question')}**\n")
        options = checkpoint.get("options") or []
        for i, opt in enumerate(options):
            letter = chr(65 + i)
            opt_text = opt.get("text") if isinstance(opt, dict) else str(opt)
            lines.append(f"- **{letter}.** {opt_text}")
        lines.append("")

    return "\n".join(lines).strip()


@router.post("/lesson")
async def get_lesson_content(
    payload: LessonRequestSchema,
    current_user: User = Depends(get_current_user),
    mongo_db: Any = Depends(get_mongo_db)
):
    """Generates full structured lesson content synchronously via REST API with caching."""
    try:
        content = await generate_learning_content(
            topic=payload.topic,
            concept_id=payload.concept_id,
            level=payload.level,
            pattern=payload.pattern,
            prerequisites=payload.prerequisites,
            language=payload.language or "python",
            explanation_language=payload.explanation_language or "english",
            mongo_db=mongo_db,
            force_regenerate=payload.force_regenerate
        )
        if isinstance(content, dict):
            content["markdown"] = build_lesson_markdown(content)
        return content
    except Exception as e:
        logger.error(f"[get_lesson_content] Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate lesson content: {str(e)}")


@router.post("/lesson/overview")
async def get_lesson_overview_endpoint(
    payload: LessonRequestSchema,
    current_user: User = Depends(get_current_user),
    mongo_db: Any = Depends(get_mongo_db)
):
    """Generates Part 1 (Intuition, Core Idea, Mechanism, Code Snippet) synchronously via REST API."""
    try:
        from app.engine.curriculum_engine import generate_lesson_overview
        content = await generate_lesson_overview(
            topic=payload.topic,
            concept_id=payload.concept_id,
            level=payload.level,
            pattern=payload.pattern,
            prerequisites=payload.prerequisites,
            language=payload.language or "python",
            explanation_language=payload.explanation_language or "english",
            mongo_db=mongo_db
        )
        if isinstance(content, dict):
            content["markdown"] = build_lesson_markdown(content)
        return content
    except Exception as e:
        logger.error(f"[get_lesson_overview_endpoint] Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate lesson overview: {str(e)}")


@router.post("/lesson/deep-dive")
async def get_lesson_deep_dive_endpoint(
    payload: LessonRequestSchema,
    current_user: User = Depends(get_current_user),
    mongo_db: Any = Depends(get_mongo_db)
):
    """Generates Part 2 (Walkthrough, Complexity, Edge Cases, Checkpoint Question) synchronously via REST API."""
    try:
        from app.engine.curriculum_engine import generate_lesson_deep_dive
        content = await generate_lesson_deep_dive(
            topic=payload.topic,
            concept_id=payload.concept_id,
            level=payload.level,
            pattern=payload.pattern,
            prerequisites=payload.prerequisites,
            language=payload.language or "python",
            explanation_language=payload.explanation_language or "english",
            mongo_db=mongo_db
        )
        if isinstance(content, dict):
            content["markdown"] = build_lesson_markdown(content)
        return content
    except Exception as e:
        logger.error(f"[get_lesson_deep_dive_endpoint] Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate lesson deep dive: {str(e)}")


def _is_valid_lesson_content(content: dict) -> bool:
    if not isinstance(content, dict):
        return False
    intuition = str(content.get("intuition", ""))
    mechanism = str(content.get("mechanism", ""))
    code = str(content.get("code_snippet", ""))
    if "Core intuition for this topic" in intuition or "Step-by-step mechanism details" in mechanism or "# implementation here" in code:
        return False
    return True


async def stream_lesson_generator(payload: LessonRequestSchema, mongo_db: Any):
    from app.engine.curriculum_engine import normalize_topic, shuffle_checkpoint_options
    from app.llm.gateway import call_llm_gateway_stream, parse_json_from_llm_response
    from app.llm.prompts import LEARNING_CONTENT_PROMPT
    from app.llm.schemas import LessonContentSchema
    from app.db.redis_client import get_redis_client

    norm = normalize_topic(payload.topic)
    pattern_clean = payload.pattern or "none"
    lang_clean = (payload.language or "python").lower().strip()
    exp_lang_clean = (payload.explanation_language or "english").lower().strip()
    cache_key = f"lesson_v4:{norm['slug']}:{payload.concept_id}:{payload.level}:{pattern_clean}:{lang_clean}:{exp_lang_clean}"

    # 1. Emit message_start event
    yield format_sse(MessageStartEvent(
        role="assistant",
        metadata={
            "concept_id": payload.concept_id,
            "topic": norm["canonical_topic"],
            "level": payload.level,
            "language": lang_clean
        }
    ))

    # Helper to construct typed user_input event for the Checkpoint Question
    def build_checkpoint_user_input(content_dict: dict) -> UserInputEvent:
        cp = content_dict.get("checkpoint") or {}
        raw_opts = cp.get("options") or ["Option A", "Option B", "Option C", "Option D"]
        input_opts = []
        for i, opt in enumerate(raw_opts):
            letter = chr(65 + i)
            opt_text = opt.get("text") if isinstance(opt, dict) else str(opt)
            input_opts.append(InputOption(id=letter, label=opt_text, value=opt_text))

        corr = cp.get("correct_option") or cp.get("correct_option_id") or (input_opts[0].label if input_opts else "Option A")
        expl = cp.get("explanation") or "Validates concept mechanism."
        q_text = cp.get("question") or f"Which statement best describes the operational mechanism of {payload.concept_id}?"

        u_input = UserInputPayload(
            input_id=f"checkpoint_{payload.concept_id}",
            type="single_select",
            label="Interactive Concept Checkpoint",
            description=q_text,
            options=input_opts,
            required=True,
            validation=ValidationConstraints(options=[o.label for o in input_opts]),
            metadata={
                "concept_id": payload.concept_id,
                "correct_option": corr,
                "explanation": expl
            }
        )
        return UserInputEvent(input=u_input)

    # Helper to stream Markdown text in progressive chunks
    def stream_md_chunks(md_text: str):
        chunk_size = 120
        events = []
        for i in range(0, len(md_text), chunk_size):
            events.append(format_sse(TextDeltaEvent(delta=md_text[i:i + chunk_size])))
        return events

    # 1. Check Redis & MongoDB Cache (bypassed if force_regenerate=True)
    redis_c = get_redis_client()
    cached_content = None

    if payload.force_regenerate:
        logger.info(f"[stream_lesson_generator] force_regenerate=True -> Invalidation cache for '{cache_key}'")
        if redis_c:
            try:
                await redis_c.delete(cache_key)
            except Exception:
                pass
        if mongo_db is not None:
            try:
                lessons_col = mongo_db.get_collection("lessons")
                await lessons_col.delete_one({"_id": cache_key})
            except Exception:
                pass
    else:
        if redis_c:
            try:
                cached_data = await redis_c.get(cache_key)
                if cached_data:
                    cached_content = shuffle_checkpoint_options(json.loads(cached_data))
                    if not _is_valid_lesson_content(cached_content):
                        await redis_c.delete(cache_key)
                        cached_content = None
            except Exception:
                pass

        if not cached_content and mongo_db is not None:
            try:
                lessons_col = mongo_db.get_collection("lessons")
                existing_doc = await lessons_col.find_one({"_id": cache_key})
                if existing_doc and "content" in existing_doc:
                    cached_content = shuffle_checkpoint_options(existing_doc["content"])
                    if not _is_valid_lesson_content(cached_content):
                        await lessons_col.delete_one({"_id": cache_key})
                        cached_content = None
            except Exception:
                pass

    if cached_content:
        full_md = cached_content.get("markdown") or build_lesson_markdown(cached_content)
        cached_content["markdown"] = full_md

        yield format_sse(ToolStartEvent(
            tool_name="lesson_generator",
            tool_input={"concept_id": payload.concept_id, "topic": norm["canonical_topic"], "level": payload.level, "cached": True}
        ))

        # Stream pure Markdown chunks
        for evt_str in stream_md_chunks(full_md):
            yield evt_str

        yield format_sse(ToolEndEvent(
            tool_name="lesson_generator",
            tool_output={"concept_id": payload.concept_id, "cached": True}
        ))

        # Emit tool call for question generation
        yield format_sse(ToolStartEvent(
            tool_name="generate_question",
            tool_input={"concept_id": payload.concept_id, "topic": norm["canonical_topic"], "type": "mcq", "cached": True}
        ))
        yield format_sse(ToolEndEvent(
            tool_name="generate_question",
            tool_output={"concept_id": payload.concept_id, "checkpoint": cached_content.get("checkpoint")}
        ))

        cp_cached = cached_content.get("checkpoint") or {}
        yield f"data: {json.dumps({'event': 'question', 'question': cp_cached.get('question'), 'options': cp_cached.get('options'), 'concept_id': payload.concept_id, 'type': 'mcq'})}\n\n"

        # Emit structured user_input event for interactive Checkpoint Question
        yield format_sse(build_checkpoint_user_input(cached_content))

        # Emit message_end event awaiting user
        yield format_sse(MessageEndEvent(
            status="awaiting_user",
            payload={
                "content": cached_content,
                "markdown": full_md,
                "concept_id": payload.concept_id,
                "cached": True
            }
        ))
        # Legacy event for backward compatibility
        yield f"data: {json.dumps({'event': 'cached', 'content': cached_content, 'markdown': full_md})}\n\n"
        return

    # 3. Cache MISS -> Stream live markdown tokens from LLM
    yield format_sse(ToolStartEvent(
        tool_name="lesson_generator",
        tool_input={"concept_id": payload.concept_id, "topic": norm["canonical_topic"], "level": payload.level}
    ))
    yield f"data: {json.dumps({'event': 'start', 'status': 'generating'})}\n\n"

    prereq_str = str(payload.prerequisites) if payload.prerequisites else "None"
    full_tokens = []

    try:
        from app.llm.prompts import LESSON_MARKDOWN_PROMPT
        async for chunk_text in call_llm_gateway_stream(
            LESSON_MARKDOWN_PROMPT,
            {
                "topic": norm["canonical_topic"],
                "concept_id": payload.concept_id,
                "level": payload.level,
                "pattern": pattern_clean,
                "prerequisites": prereq_str,
                "language": lang_clean,
                "explanation_language": exp_lang_clean
            }
        ):
            full_tokens.append(chunk_text)
            # Live Markdown stream to frontend as text_delta event
            yield format_sse(TextDeltaEvent(delta=chunk_text))

        raw_llm_str = "".join(full_tokens)
        extracted_checkpoint = None

        # Check if the output was structured JSON (e.g. from mock or json-constrained model)
        if raw_llm_str.strip().startswith("{") and '"checkpoint"' in raw_llm_str:
            try:
                parsed = parse_json_from_llm_response(raw_llm_str, LessonContentSchema, topic_hint=payload.concept_id)
                content_dict = parsed.model_dump() if hasattr(parsed, "model_dump") else dict(parsed)
                extracted_checkpoint = content_dict.get("checkpoint")
                full_md = build_lesson_markdown(content_dict)
            except Exception:
                full_md = raw_llm_str
        else:
            full_md = raw_llm_str.strip()

        # If full_md is valid markdown and full_tokens was JSON that didn't stream clean markdown
        if raw_llm_str.strip().startswith("{") and full_md != raw_llm_str:
            for evt_str in stream_md_chunks(full_md):
                yield evt_str

        # End of lesson markdown generation tool
        yield format_sse(ToolEndEvent(
            tool_name="lesson_generator",
            tool_output={"concept_id": payload.concept_id, "cached": False}
        ))

        # 4. Generate Checkpoint MCQ Question Tool Call
        yield format_sse(ToolStartEvent(
            tool_name="generate_question",
            tool_input={
                "topic": norm["canonical_topic"],
                "concept_id": payload.concept_id,
                "level": payload.level,
                "type": "mcq"
            }
        ))
        yield f"data: {json.dumps({'event': 'question_start', 'tool': 'generate_question', 'concept_id': payload.concept_id})}\n\n"

        checkpoint_data = extracted_checkpoint
        if not checkpoint_data or not isinstance(checkpoint_data, dict) or not checkpoint_data.get("question"):
            try:
                from app.engine.question_engine import get_or_generate_question
                diff_val = 2 if (payload.level or "beginner").lower() == "beginner" else 3 if (payload.level or "").lower() == "intermediate" else 4
                q_doc = await get_or_generate_question(
                    topic=norm["canonical_topic"],
                    concept_id=payload.concept_id,
                    concept_type="conceptual",
                    difficulty=diff_val,
                    language=lang_clean,
                    explanation_language=exp_lang_clean,
                    mongo_db=mongo_db,
                    include_internal=True
                )
                raw_opts = q_doc.get("options") or ["Option A", "Option B", "Option C", "Option D"]
                checkpoint_data = {
                    "question": q_doc.get("question") or f"Which statement best characterizes {payload.concept_id.replace('_', ' ').title()}?",
                    "options": raw_opts,
                    "correct_option": q_doc.get("correct_option") or raw_opts[0],
                    "explanation": q_doc.get("explanation") or q_doc.get("rubric") or "Validates concept mechanism."
                }
            except Exception as q_err:
                logger.warning(f"[stream_lesson_generator] Question generation tool fallback: {q_err}")
                checkpoint_data = {
                    "question": f"Which statement best describes the operational mechanism and trade-offs of {payload.concept_id.replace('_', ' ').title()}?",
                    "options": [
                        f"It operates with O(1) constant access by maintaining concept invariants.",
                        f"It requires O(n) traversal for every basic operation.",
                        f"It cannot dynamically allocate memory or handle resizing.",
                        f"It invalidates downstream pointers during operations."
                    ],
                    "correct_option": f"It operates with O(1) constant access by maintaining concept invariants.",
                    "explanation": f"Validates fundamental invariant understanding of {payload.concept_id.replace('_', ' ').title()}."
                }

        checkpoint_data = shuffle_checkpoint_options({"checkpoint": checkpoint_data})["checkpoint"]

        # Tool end for question generation with clean structured JSON
        yield format_sse(ToolEndEvent(
            tool_name="generate_question",
            tool_output={
                "concept_id": payload.concept_id,
                "question": checkpoint_data.get("question"),
                "options": checkpoint_data.get("options"),
                "type": "mcq"
            }
        ))

        # Explicit 'question' event with clean JSON structure for UI rendering
        yield f"data: {json.dumps({'event': 'question', 'question': checkpoint_data.get('question'), 'options': checkpoint_data.get('options'), 'concept_id': payload.concept_id, 'type': 'mcq'})}\n\n"

        # Emit user_input event with the interactive Checkpoint Question
        yield format_sse(build_checkpoint_user_input({"checkpoint": checkpoint_data}))

        # Cache content with markdown & checkpoint
        content = {
            "concept_id": payload.concept_id,
            "topic": norm["canonical_topic"],
            "level": payload.level,
            "markdown": full_md,
            "checkpoint": checkpoint_data
        }
        if redis_c:
            try:
                await redis_c.set(cache_key, json.dumps(content), ex=86400)
            except Exception:
                pass
        if mongo_db is not None:
            try:
                lessons_col = mongo_db.get_collection("lessons")
                await lessons_col.update_one(
                    {"_id": cache_key},
                    {"$set": {"_id": cache_key, "topic": norm["slug"], "concept_id": payload.concept_id, "language": lang_clean, "content": content}},
                    upsert=True
                )
            except Exception:
                pass

        # Emit message_end event awaiting user interaction on checkpoint
        yield format_sse(MessageEndEvent(
            status="awaiting_user",
            payload={
                "content": content,
                "markdown": full_md,
                "concept_id": payload.concept_id,
                "checkpoint": checkpoint_data
            }
        ))
        yield f"data: {json.dumps({'event': 'complete', 'content': content, 'markdown': full_md, 'checkpoint': checkpoint_data})}\n\n"

    except Exception as err:
        logger.error(f"[stream_lesson_generator] Error streaming lesson: {err}", exc_info=True)
        yield format_sse(ErrorEvent(message=f"Lesson generation failed: {str(err)}", code="LESSON_STREAM_ERROR"))
        yield f"data: {json.dumps({'event': 'error', 'message': str(err)})}\n\n"
        return




@router.post("/lesson/stream")
async def get_lesson_content_stream(
    payload: LessonRequestSchema,
    current_user: User = Depends(get_current_user),
    mongo_db: Any = Depends(get_mongo_db)
):
    """
    Server-Sent Events (SSE) streaming endpoint for interactive lesson generation.
    Streams token deltas live as LLM generates structured lesson text.
    """
    return StreamingResponse(
        stream_lesson_generator(payload, mongo_db),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@router.post("/lesson/regenerate/stream")
@router.post("/lesson/regenerate")
async def regenerate_lesson_stream(
    payload: LessonRequestSchema,
    current_user: User = Depends(get_current_user),
    mongo_db: Any = Depends(get_mongo_db)
):
    """
    Dedicated endpoint to force-regenerate a brand-new lesson and checkpoint question
    for the exact same concept node, completely bypassing and replacing existing cache.
    """
    # Ensure force_regenerate is explicitly True
    payload.force_regenerate = True
    return StreamingResponse(
        stream_lesson_generator(payload, mongo_db),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@router.post("/lesson/checkpoint-answer")
async def answer_checkpoint(
    payload: CheckpointAnswerSchema,
    current_user: User = Depends(get_current_user),
    mongo_db: Any = Depends(get_mongo_db)
):
    """
    Validates and evaluates student's response to the interactive Checkpoint Question (user_input event).
    Associates answer with concept_id and input_id, validating against options.
    """
    topic_slug = slugify_topic(payload.topic)
    concept_id = payload.concept_id.strip().lower()

    # Retrieve lesson from MongoDB or Redis to check correct answer
    lesson_doc = None
    if mongo_db:
        lessons_col = mongo_db.get_collection("lessons")
        lesson_doc = await lessons_col.find_one({"concept_id": concept_id, "topic": topic_slug})
        if not lesson_doc:
            lesson_doc = await lessons_col.find_one({"concept_id": concept_id})

    correct_option = "Option A"
    explanation = "Validates core concept mechanism."
    options = []
    if lesson_doc and "content" in lesson_doc:
        cp = lesson_doc["content"].get("checkpoint") or {}
        correct_option = cp.get("correct_option") or cp.get("correct_option_id") or "Option A"
        explanation = cp.get("explanation") or explanation
        options = cp.get("options") or []

    # Validate against input schema if options known
    clean_ans = str(payload.answer).strip()
    clean_corr = str(correct_option).strip()

    if options:
        opt_labels = [o.get("text") if isinstance(o, dict) else str(o) for o in options]
        temp_input = UserInputPayload(
            input_id=payload.input_id,
            type="single_select",
            label="Interactive Concept Checkpoint",
            options=[InputOption(id=chr(65+i), label=lbl, value=lbl) for i, lbl in enumerate(opt_labels)],
            required=True
        )
        is_val, val_err = validate_user_input_submission(temp_input, clean_ans)
        if not is_val:
            raise HTTPException(status_code=400, detail=val_err or "Invalid selection")

    is_correct = (
        clean_ans.lower() == clean_corr.lower() or
        clean_ans.lower() in clean_corr.lower() or
        clean_corr.lower() in clean_ans.lower()
    )

    return {
        "concept_id": concept_id,
        "input_id": payload.input_id,
        "is_correct": is_correct,
        "submitted_answer": payload.answer,
        "correct_option": clean_corr,
        "explanation": explanation,
        "feedback": "Correct! Excellent understanding of the concept." if is_correct else f"Incorrect. The correct answer is: {clean_corr}."
    }


@router.get("/{user_id}/{topic}")
async def get_learning_path(
    user_id: str,
    topic: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    mongo_db: Any = Depends(get_mongo_db)
):
    """Retrieves current personalized learning path for user and topic."""
    target_user_id = str(current_user.id)
    slug = slugify_topic(topic)
    
    path_doc = await sync_learning_path_progress(
        user_id=target_user_id,
        topic_slug=slug,
        mongo_db=mongo_db,
        db=db
    )

    if not path_doc:
        lp_collection = mongo_db.get_collection("learning_paths")
        path_doc = await lp_collection.find_one({
            "user_id": target_user_id,
            "topic": slug
        })
        if not path_doc:
            path_doc = await lp_collection.find_one({
                "user_id": target_user_id,
                "topic": topic.lower()
            })

    if not path_doc:
        raise HTTPException(status_code=404, detail="Learning path not found for given user and topic.")

    return path_doc

@router.post("/reteach")
async def reteach_concept(
    payload: ReteachRequestSchema,
    current_user: User = Depends(get_current_user)
):
    """Executes stateful LangGraph adaptive re-teaching flow to try a new pedagogical strategy."""
    state = {
        "user_id": str(current_user.id),
        "concept_id": payload.concept_id,
        "topic": payload.topic,
        "strategies_tried": payload.strategies_tried,
        "current_strategy": None,
        "explanation_content": None,
        "consecutive_failures": payload.consecutive_failures,
        "action": "continue"
    }

    result_state = await select_strategy_node(state)
    return result_state


class PracticeVerifiedPayload(BaseModel):
    user_id: str
    topic: str
    concept_id: str
    passed: bool


@router.post("/verify-practice")
async def verify_practice(payload: PracticeVerifiedPayload, db: AsyncSession = Depends(get_db)):
    """Updates concept mastery once client validates passing test cases."""
    try:
        user_uuid = uuid.UUID(payload.user_id) if isinstance(payload.user_id, str) else payload.user_id
    except ValueError:
        user_uuid = payload.user_id

    topic_slug = slugify_topic(payload.topic)
    concept_id_val = payload.concept_id.strip().lower()

    stmt = select(ConceptMastery).where(
        ConceptMastery.user_id == user_uuid,
        ConceptMastery.topic == topic_slug,
        ConceptMastery.concept_id == concept_id_val
    )
    res = await db.execute(stmt)
    mastery = res.scalar_one_or_none()

    if mastery:
        mastery.attempts += 1
        if payload.passed:
            # Exponential Moving Average update
            mastery.mastery = min(1.0, mastery.mastery + 0.25)
            if mastery.mastery >= 0.8:
                mastery.status = "known"
        else:
            mastery.failures += 1
            mastery.mastery = max(0.0, mastery.mastery - 0.15)
        await db.commit()
    else:
        new_mastery_val = 0.25 if payload.passed else 0.0
        status_val = "known" if (payload.passed and new_mastery_val >= 0.8) else "learning"
        mastery = ConceptMastery(
            user_id=user_uuid,
            topic=topic_slug,
            concept_id=concept_id_val,
            mastery=new_mastery_val,
            attempts=1,
            failures=0 if payload.passed else 1,
            status=status_val
        )
        db.add(mastery)
        await db.commit()

    return {"status": "success", "new_mastery": mastery.mastery if mastery else 0.0}

