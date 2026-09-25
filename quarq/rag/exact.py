"""Exact (brute-force) nearest-neighbour search over a VectorStore's embeddings.

ChromaDB's HNSW index is approximate: it can leave out chunks that score higher
than ones it returns. ExactStore scores every stored chunk, so it is the reference
the eval compares the index against. It has VectorStore.query's interface, so the
normal Retriever can run on top of it unchanged.

It holds every embedding in memory as one float32 array (4,235 x 1,024 is about
17 MB), so it is meant for evaluation, not for serving queries.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from quarq.exceptions import RAGError
from quarq.rag.store import RetrievedChunk

if TYPE_CHECKING:
    from quarq.rag.store import VectorStore


class ExactStore:
    """Brute-force cosine search over a snapshot of a VectorStore.

    Args:
        store: The store whose chunks to snapshot. Later changes to it are not seen.

    Raises:
        RAGError: If the store's records can't be read.
    """

    def __init__(self, store: VectorStore) -> None:
        records = store.all_records()
        self._ids: list[str] = records["ids"]
        self._documents: list[str] = records["documents"]
        self._metadatas: list[dict[str, Any]] = records["metadatas"]
        matrix = np.asarray(records["embeddings"], dtype=np.float32)
        if matrix.size:
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            matrix /= np.where(norms == 0, 1.0, norms)  # in place: no second copy
        self._matrix = matrix

    def count(self) -> int:
        """Return the number of chunks in the snapshot.

        Returns:
            Chunk count.
        """
        return len(self._ids)

    def query(
        self,
        embedding: list[float],
        k: int = 5,
        filters: dict | None = None,
    ) -> list[RetrievedChunk]:
        """Return the k chunks with the highest cosine similarity to the embedding.

        Args:
            embedding: Query vector.
            k: Maximum number of results.
            filters: Optional metadata equality filters, e.g. {'doc_type': 'ecb_fsr'}.

        Returns:
            Chunks sorted by similarity, highest first.

        Raises:
            RAGError: If a filter is not a plain equality.
        """
        if not self._ids:
            return []
        mask = np.ones(len(self._ids), dtype=bool)
        for key, value in (filters or {}).items():
            if isinstance(value, dict) or key.startswith("$"):
                raise RAGError(f"ExactStore supports equality filters only, got {key}={value!r}")
            mask &= np.array([meta.get(key) == value for meta in self._metadatas])

        query = np.asarray(embedding, dtype=np.float32)
        norm = np.linalg.norm(query)
        sims = self._matrix @ (query / norm if norm else query)
        candidates = np.flatnonzero(mask)
        order = candidates[np.argsort(-sims[candidates], kind="stable")][:k]
        return [
            RetrievedChunk(
                content=self._documents[i],
                metadata=self._metadatas[i],
                similarity=float(sims[i]),
                source=str(self._metadatas[i].get("source", "")),
                page=int(self._metadatas[i].get("page", 0)),
            )
            for i in order
        ]
