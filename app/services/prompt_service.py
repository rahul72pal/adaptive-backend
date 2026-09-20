from typing import Optional

DEFAULT_AGENT_SYSTEM_PROMPT = """You are an expert adaptive learning AI tutor specializing in Data Structures & Algorithms (DSA).
Your role is to guide students through concept learning, interactive problem solving, diagnostic probing, and clear conceptual explanations.
Provide clear, structured, and pedagogical responses. Use available search tools when factual verification is needed."""

class PromptService:
    def __init__(self, default_prompt: Optional[str] = None):
        self._default_prompt = default_prompt or DEFAULT_AGENT_SYSTEM_PROMPT

    def get_system_prompt(self, behavior: Optional[str] = None) -> str:
        if behavior and behavior.strip():
            return behavior.strip()
        return self._default_prompt
