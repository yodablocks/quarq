"""quarq LLM backend layer."""

from quarq.llm.base import BaseLLM, GenerationResult
from quarq.llm.claude import ClaudeLLM
from quarq.llm.lmstudio import LMStudioLLM, get_llm, is_available

__all__ = [
    "BaseLLM",
    "GenerationResult",
    "ClaudeLLM",
    "LMStudioLLM",
    "get_llm",
    "is_available",
]
