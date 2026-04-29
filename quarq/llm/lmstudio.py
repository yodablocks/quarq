"""LM Studio OpenAI-compatible local LLM backend."""

from __future__ import annotations

import openai

from quarq.config import QuarqConfig
from quarq.exceptions import RAGError
from quarq.llm.base import BaseLLM

_VALID_AGENTS = ("reporting", "research")
_AVAILABILITY_TIMEOUT = 3.0
_GENERATION_TIMEOUT = 60.0


class LMStudioLLM(BaseLLM):
    """OpenAI-compatible backend targeting a local LM Studio server.

    Args:
        cfg: Loaded QuarqConfig.
        agent: Either 'reporting' or 'research'. Determines which model is used.

    Raises:
        RAGError: If agent is not 'reporting' or 'research'.
    """

    def __init__(self, cfg: QuarqConfig, agent: str = "research") -> None:
        if agent not in _VALID_AGENTS:
            raise RAGError(f"agent must be one of {_VALID_AGENTS}, got {agent!r}")
        self._cfg = cfg
        self._agent = agent
        self._client = openai.OpenAI(
            base_url=cfg.lmstudio.url,
            api_key="not-required",
        )
        self._discovered_model: str | None = None

    @property
    def name(self) -> str:  # type: ignore[override]
        """Backend identifier."""
        return "lmstudio"

    @property
    def model(self) -> str:
        """Active model name — discovered from LM Studio if auto_discover_model is set."""
        if self._cfg.lmstudio.auto_discover_model:
            if self._discovered_model is None:
                self._discovered_model = self._discover_model()
            return self._discovered_model
        if self._agent == "reporting":
            return self._cfg.llm.reporting_model
        return self._cfg.llm.research_model

    def _discover_model(self) -> str:
        """Query LM Studio for the first available model ID.

        Returns:
            The first model ID returned by LM Studio, or the config model name on failure.
        """
        fallback = (
            self._cfg.llm.reporting_model
            if self._agent == "reporting"
            else self._cfg.llm.research_model
        )
        try:
            models = self._client.models.list()
            first = next(iter(models.data), None)
            return first.id if first is not None else fallback
        except Exception:
            return fallback

    def generate(self, prompt: str, system: str = "") -> str:
        """Send a prompt to LM Studio and return the response text.

        Args:
            prompt: User message text.
            system: Optional system prompt.

        Returns:
            Generated text string.

        Raises:
            RAGError: If the server is unreachable or returns a malformed response.
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,  # type: ignore[arg-type]
                timeout=_GENERATION_TIMEOUT,
            )
            content = response.choices[0].message.content
            if content is None:
                raise RAGError("LM Studio returned empty message content")
            return content
        except RAGError:
            raise
        except Exception as exc:
            raise RAGError(f"LM Studio generation failed: {exc}") from exc


def is_available(cfg: QuarqConfig) -> bool:
    """Check whether LM Studio is reachable within 3 seconds.

    Args:
        cfg: Loaded QuarqConfig.

    Returns:
        True if LM Studio responds to a model listing request.
    """
    try:
        client = openai.OpenAI(
            base_url=cfg.lmstudio.url,
            api_key="not-required",
            timeout=_AVAILABILITY_TIMEOUT,
        )
        client.models.list()
        return True
    except Exception:
        return False


def get_llm(cfg: QuarqConfig, agent: str = "research") -> BaseLLM:
    """Return the best available LLM backend for the given agent role.

    Tries LM Studio first; falls back to Claude API if unavailable.

    Args:
        cfg: Loaded QuarqConfig.
        agent: 'reporting' or 'research'.

    Returns:
        A BaseLLM instance ready to use.

    Raises:
        RAGError: If neither LM Studio nor the Claude API is available.
    """
    if is_available(cfg):
        return LMStudioLLM(cfg, agent=agent)

    # Fall back to Claude API
    import os

    if os.environ.get("ANTHROPIC_API_KEY"):
        from quarq.llm.claude import ClaudeLLM
        return ClaudeLLM(cfg, agent=agent)

    raise RAGError("No LLM backend available")
