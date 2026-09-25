"""Abstract base class and result types for all LLM backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class GenerationResult:
    """Result of a single LLM generation call.

    Attributes:
        answer: The generated text response.
        model: Model identifier used for generation.
        backend: Backend name (e.g. 'lmstudio', 'claude').
        latency_ms: Wall-clock time for the call in milliseconds.
        figures_checked: For research answers, how many figures were checked against
            the prompt; None when no check ran.
        unsupported_figures: Figures in the answer that appear nowhere in the prompt.
    """

    answer: str
    model: str
    backend: str
    latency_ms: int
    figures_checked: int | None = None
    unsupported_figures: list[str] | None = None


class BaseLLM(ABC):
    """Contract every LLM backend must implement.

    Two agents are defined — reporting (fast) and research (heavy).
    Never use a hardcoded model name; always read from config.
    """

    @abstractmethod
    def generate(self, prompt: str, system: str = "") -> str:
        """Generate a text response from the given prompt.

        Args:
            prompt: The user-facing prompt text.
            system: Optional system instruction string.

        Returns:
            Generated text as a plain string.

        Raises:
            RAGError: On any generation failure.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Short lowercase backend identifier (e.g. 'lmstudio', 'claude')."""
        ...

    @property
    @abstractmethod
    def model(self) -> str:
        """Model identifier string as configured."""
        ...
