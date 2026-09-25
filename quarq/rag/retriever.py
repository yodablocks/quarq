"""Top-k retrieval with similarity filtering and optional metadata constraints."""

from __future__ import annotations

import logging

from quarq.config import QuarqConfig
from quarq.constants import RETRIEVAL_CANDIDATE_POOL, RETRIEVAL_OVERFETCH_FACTOR
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
        self._corpus_ready: bool | None = None

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
            Up to k chunks sorted by similarity descending, at most one per
            (source, page): the best-scoring chunk of each page. Returns [] if
            no chunks meet the threshold.
        """
        effective_k = k if k is not None else self._cfg.rag.top_k
        threshold = min_similarity if min_similarity is not None else self._cfg.rag.min_similarity

        if self._corpus_ready is None:
            try:
                self._corpus_ready = int(self._store.count()) > 0
            except (TypeError, ValueError):
                self._corpus_ready = True

        query_embedding = self._embedder.embed_query(query)

        filters = {"doc_type": doc_type} if doc_type else None
        # Ask for a wide candidate pool: HNSW recall depends on how many results are requested.
        n_candidates = max(effective_k * RETRIEVAL_OVERFETCH_FACTOR, RETRIEVAL_CANDIDATE_POOL)
        chunks = self._store.query(query_embedding, k=n_candidates, filters=filters)

        filtered = [c for c in chunks if c.similarity >= threshold]
        filtered.sort(key=lambda c: c.similarity, reverse=True)
        filtered = _best_chunk_per_page(filtered)[:effective_k]

        if not filtered:
            logger.debug(
                "No chunks above min_similarity=%.2f for query: %s", threshold, query[:80]
            )

        return filtered

    def invalidate_corpus_cache(self) -> None:
        """Reset the cached corpus-ready flag so the next retrieve() re-checks count."""
        self._corpus_ready = None


def _best_chunk_per_page(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Keep the first chunk seen for each (source, page), preserving order.

    Args:
        chunks: Chunks already sorted by similarity, best first.

    Returns:
        The chunks with later same-page duplicates removed.
    """
    seen: set[tuple[str, int]] = set()
    unique: list[RetrievedChunk] = []
    for chunk in chunks:
        key = (chunk.source, chunk.page)
        if key not in seen:
            seen.add(key)
            unique.append(chunk)
    return unique
