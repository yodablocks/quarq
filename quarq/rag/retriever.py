"""Top-k retrieval with similarity filtering and optional metadata constraints."""

from __future__ import annotations

import logging

from quarq.config import QuarqConfig
from quarq.rag.embedder import Embedder
from quarq.rag.store import RetrievedChunk, VectorStore

logger = logging.getLogger(__name__)


class Retriever:
    """Retrieves relevant document chunks for a query using vector similarity.

    Args:
        store: Initialised VectorStore instance.
        embedder: Initialised Embedder instance.
        cfg: Loaded QuarqConfig (provides default k and min_similarity).
    """

    def __init__(self, store: VectorStore, embedder: Embedder, cfg: QuarqConfig) -> None:
        self._store = store
        self._embedder = embedder
        self._cfg = cfg

    def retrieve(
        self,
        query: str,
        k: int | None = None,
        doc_type: str | None = None,
        min_similarity: float | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve the most relevant chunks for a query.

        Args:
            query: Natural language query string.
            k: Maximum chunks to return. Defaults to config.rag.top_k.
            doc_type: Optional metadata filter — restrict to a specific doc_type.
            min_similarity: Minimum cosine similarity threshold.
                            Defaults to config.rag.min_similarity.

        Returns:
            List of RetrievedChunk sorted by similarity descending.
            Returns [] if no chunks meet the threshold.
        """
        effective_k = k if k is not None else self._cfg.rag.top_k
        threshold = min_similarity if min_similarity is not None else self._cfg.rag.min_similarity

        query_embedding = self._embedder.embed_query(query)

        filters = {"doc_type": doc_type} if doc_type else None
        chunks = self._store.query(query_embedding, k=effective_k, filters=filters)

        filtered = [c for c in chunks if c.similarity >= threshold]
        filtered.sort(key=lambda c: c.similarity, reverse=True)

        if not filtered:
            logger.debug(
                "No chunks above min_similarity=%.2f for query: %s", threshold, query[:80]
            )

        return filtered
