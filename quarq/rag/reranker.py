"""Cross-encoder re-ranking of retrieved chunks.

The embedder scores a question and a passage separately, so it can rank a summary page
or another edition above the page that holds the answer. A cross-encoder reads the
question and the passage together and re-orders the top candidates. On the gold set it
raised the right page at rank 1 from 22 to 32 of 38 questions.

The model is loaded lazily on first use, so constructing a Reranker is cheap.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any

from quarq.exceptions import RAGError
from quarq.rag.store import RetrievedChunk

logger = logging.getLogger(__name__)


def _best_device() -> str:
    """Pick the fastest available torch device.

    Returns:
        "cuda", "mps" (Apple GPU) or "cpu".
    """
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Reranker:
    """Re-orders chunks by a cross-encoder's relevance to the question.

    Args:
        model_name: Hugging Face model id, from config (rag.reranker_model).
        max_length: Maximum tokens per question + passage pair.
    """

    def __init__(self, model_name: str, max_length: int) -> None:
        self._model_name = model_name
        self._max_length = max_length
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(
                    self._model_name, max_length=self._max_length, device=_best_device()
                )
            except Exception as exc:
                raise RAGError(f"Failed to load re-ranker {self._model_name!r}: {exc}") from exc
        return self._model

    def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Score each chunk against the question and return them best first.

        Args:
            query: The question.
            chunks: Candidate chunks.

        Returns:
            New RetrievedChunk objects with rerank_score set, sorted by it (highest first).

        Raises:
            RAGError: If the model can't be loaded or scoring fails.
        """
        if not chunks:
            return []
        model = self._load()
        try:
            scores = model.predict([(query, c.content) for c in chunks])
        except Exception as exc:
            raise RAGError(f"Re-ranking failed: {exc}") from exc
        scored = [replace(c, rerank_score=float(s)) for c, s in zip(chunks, scores, strict=True)]
        return sorted(scored, key=lambda c: c.rerank_score or 0.0, reverse=True)
