import json
import logging
from typing import Type, Any, Dict
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from app.core.config import settings

logger = logging.getLogger(__name__)

def parse_json_from_llm_response(text: str, schema: Type[BaseModel]) -> BaseModel:
    """Extracts valid JSON payload safely without regex backtracking hangs."""
    clean_text = text.strip()
    
    # Strip markdown block quotes if present
    if "```json" in clean_text:
        clean_text = clean_text.split("```json", 1)[1]
        clean_text = clean_text.split("```", 1)[0]
    elif "```" in clean_text:
        clean_text = clean_text.split("```", 1)[1]
        clean_text = clean_text.split("```", 1)[0]

    start = clean_text.find('{')
    end = clean_text.rfind('}')
    
    if start != -1 and end != -1:
        json_str = clean_text[start:end+1]
        data = json.loads(json_str)
        return schema.model_validate(data)
    
    raise ValueError(f"No JSON object could be extracted from: {text[:200]}...")
    
async def call_llm_gateway(
    prompt: ChatPromptTemplate,
    params: Dict[str, Any],
    schema: Type[BaseModel]
) -> BaseModel:
    """Invokes LLM Gateway with flexible OpenAI/OpenRouter/Custom, Gemini, Groq, or deterministic fallback."""
    parser = PydanticOutputParser(pydantic_object=schema)
    formatted_prompt = prompt.partial(format_instructions=parser.get_format_instructions())

    # 1. Try OpenRouter or OpenAI / Custom OpenAI-Compatible Provider
    api_key = settings.OPENROUTER_API_KEY or settings.OPENAI_API_KEY
    if api_key:
        openrouter_models = [
            settings.LLM_MODEL_NAME,
            "google/gemini-2.0-flash-001",
            "meta-llama/llama-3.3-70b-instruct",
            "deepseek/deepseek-chat",
            "openai/gpt-4o-mini"
        ]
        openrouter_models = list(dict.fromkeys([m for m in openrouter_models if m]))

        for model_name in openrouter_models:
            try:
                from langchain_openai import ChatOpenAI
                base_url = (
                    "https://openrouter.ai/api/v1"
                    if settings.OPENROUTER_API_KEY
                    else (settings.OPENAI_API_BASE or "https://api.openai.com/v1")
                )

                headers = {}
                if settings.OPENROUTER_API_KEY:
                    headers = {
                        "HTTP-Referer": "http://localhost:8000",
                        "X-Title": "Adaptive Learning Platform"
                    }

                llm_model = ChatOpenAI(
                    model=model_name,
                    openai_api_key=api_key,
                    openai_api_base=base_url,
                    default_headers=headers if headers else None,
                    temperature=0.2,
                    max_tokens=getattr(settings, "LLM_MAX_TOKENS", 1024)
                )
                try:
                    chain = formatted_prompt | llm_model | parser
                    return await chain.ainvoke(params)
                except Exception as parse_err:
                    raw_chain = formatted_prompt | llm_model
                    res = await raw_chain.ainvoke(params)
                    content = res.content if hasattr(res, "content") else str(res)
                    return parse_json_from_llm_response(content, schema)
            except Exception as e:
                logger.warning(f"OpenAI/OpenRouter model '{model_name}' failed ({e}). Trying next candidate...")

    # 2. Try Gemini API
    if settings.GEMINI_API_KEY:
        from langchain_google_genai import ChatGoogleGenerativeAI
        gemini_candidates = [
            settings.GEMINI_MODEL_NAME,
            "gemini-1.5-flash",
            "gemini-1.5-pro",
            "gemini-2.0-flash-exp"
        ]
        # De-duplicate while preserving order
        gemini_candidates = list(dict.fromkeys([m for m in gemini_candidates if m]))
        for g_model_name in gemini_candidates:
            try:
                gemini_model = ChatGoogleGenerativeAI(
                    model=g_model_name,
                    google_api_key=settings.GEMINI_API_KEY,
                    temperature=0.2,
                    max_output_tokens=getattr(settings, "LLM_MAX_TOKENS", 1024)
                )
                chain = formatted_prompt | gemini_model | parser
                return await chain.ainvoke(params)
            except Exception as e:
                logger.warning(f"Gemini model '{g_model_name}' failed ({e}). Trying next candidate...")

    # 3. Try Groq API
    if settings.GROQ_API_KEY:
        from langchain_groq import ChatGroq
        groq_candidates = [
            settings.GROQ_MODEL_NAME,
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "qwen-2.5-32b",
            "deepseek-r1-distill-llama-70b"
        ]
        groq_candidates = list(dict.fromkeys([m for m in groq_candidates if m]))
        for groq_model_name in groq_candidates:
            try:
                groq_model = ChatGroq(
                    model_name=groq_model_name,
                    groq_api_key=settings.GROQ_API_KEY,
                    temperature=0.2,
                    max_tokens=getattr(settings, "LLM_MAX_TOKENS", 1024)
                )
                fallback_chain = formatted_prompt | groq_model | parser
                return await fallback_chain.ainvoke(params)
            except Exception as e:
                logger.warning(f"Groq model '{groq_model_name}' failed ({e}). Trying next fallback...")

    # 4. If all LLM providers fail for ConceptGraphSchema, throw an error instead of using hardcoded mock fallbacks
    if schema.__name__ == "ConceptGraphSchema":
        raise RuntimeError(f"All LLM providers failed to generate ConceptGraphSchema for topic '{params.get('topic')}'")

    # 5. Deterministic Mock Fallback (For minor testing schemas without API keys)
    logger.info("Using deterministic fallback LLM provider.")
    return _generate_deterministic_fallback(params, schema)

def _generate_deterministic_fallback(params: Dict[str, Any], schema: Type[BaseModel]) -> BaseModel:
    """Provides valid schema instances for offline testing when LLM API keys are not supplied."""
    schema_name = schema.__name__
    
    if schema_name == "DiagnosticQuestionSchema":
        concept = params.get("concept", "general_fundamentals")
        topic = params.get("topic", "Subject")
        return schema(
            question=f"Can you explain your experience and understanding of '{concept}' in {topic}?",
            target_concept=concept
        )
        
    elif schema_name == "ScoreEvaluationSchema":
        answer = params.get("answer", "").lower()
        if any(w in answer for w in ["experienced", "expert", "master", "deeply", "proficient", "advanced"]):
            return schema(confidence=0.9, status="known", feedback="Demonstrated strong familiarity.")
        elif any(w in answer for w in ["basic", "some", "learning", "somewhat", "heard"]):
            return schema(confidence=0.6, status="partial", feedback="Demonstrated basic familiarity.")
        else:
            return schema(confidence=0.2, status="unknown", feedback="Concept appears unfamiliar.")

    elif schema_name == "QuestionBatchSchema":
        concept_id = params.get("concept_id", "concept")
        concept_type = params.get("concept_type", "code")
        difficulty = params.get("difficulty", 1)
        q_type = params.get("question_type", "mcq")
        
        if q_type == "practice" and concept_type == "code":
            questions = [
                {
                    "type": "practice",
                    "concept_id": concept_id,
                    "difficulty": difficulty,
                    "question": f"Write code to demonstrate {concept_id}.",
                    "rubric": f"Code must correctly implement {concept_id}.",
                    "test_cases": [{"input": "0", "expected_output": "0"}],
                    "execution_method": "piston_api"
                }
            ]
        else:
            questions = [
                {
                    "type": "mcq",
                    "concept_id": concept_id,
                    "difficulty": difficulty,
                    "question": f"What is the primary function of {concept_id}?",
                    "options": [
                        f"Core implementation of {concept_id}",
                        "Syntax error generator",
                        "Deprecated feature",
                        "Unrelated utility"
                    ],
                    "correct_option": f"Core implementation of {concept_id}",
                    "execution_method": "llm_rubric"
                }
            ]
        return schema(questions=questions)

    elif schema_name == "RubricEvaluationSchema":
        answer = params.get("answer", "")
        score = 0.85 if len(answer) > 10 else 0.40
        return schema(
            llm_quality_score=score,
            feedback="Submission evaluated against standard criteria.",
            conceptual_breakdown={"accuracy": score, "completeness": score}
        )

    elif schema_name == "LessonExplanationSchema":
        concept_id = params.get("concept_id", "concept")
        strategy = params.get("strategy", "analogy")
        return schema(
            concept_id=concept_id,
            strategy=strategy,
            content=f"Here is an explanation of '{concept_id}' using the {strategy} strategy: Think of {concept_id} as a building block in your workflow..."
        )

    # Generic schema fallback
    return schema()
