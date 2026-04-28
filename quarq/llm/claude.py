"""Anthropic Claude API fallback LLM backend."""

from __future__ import annotations

import os

import anthropic

from quarq.config import QuarqConfig
from quarq.exceptions import RAGError
from quarq.llm.base import BaseLLM

_MAX_TOKENS = 1024
_VALID_AGENTS = ("reporting", "research")


class ClaudeLLM(BaseLLM):
    """Anthropic Claude API backend, used as fallback when LM Studio is unavailable.

    Args:
        cfg: Loaded QuarqConfig.
        agent: Either 'reporting' or 'research'. Uses config.llm.fallback_model.

    Raises:
        RAGError: On init if ANTHROPIC_API_KEY env var is not set.
    """

    def __init__(self, cfg: QuarqConfig, agent: str = "research") -> None:
        if agent not in _VALID_AGENTS:
            raise RAGError(f"agent must be one of {_VALID_AGENTS}, got {agent!r}")
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RAGError(
                "ANTHROPIC_API_KEY environment variable is not set. "
                "Set it or ensure LM Studio is running."
            )
        self._cfg = cfg
        self._agent = agent
        self._client = anthropic.Anthropic(api_key=api_key)

    @property
    def name(self) -> str:
        """Backend identifier."""
        return "claude"

    @property
    def model(self) -> str:
        """Fallback model name from config."""
        return self._cfg.llm.fallback_model

    def generate(self, prompt: str, system: str = "") -> str:
        """Send a prompt to the Claude API and return the response text.

        Args:
            prompt: User message text.
            system: Optional system prompt.

        Returns:
            Generated text string.

        Raises:
            RAGError: On any Anthropic API error.
        """
        kwargs: dict = {
            "model": self.model,
            "max_tokens": _MAX_TOKENS,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system

        try:
            response = self._client.messages.create(**kwargs)
            block = response.content[0]
            if hasattr(block, "text"):
                return block.text
            raise RAGError("Claude API returned unexpected content block type")
        except RAGError:
            raise
        except Exception as exc:
            raise RAGError(f"Claude API generation failed: {exc}") from exc
