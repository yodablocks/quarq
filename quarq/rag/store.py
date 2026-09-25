"""ChromaDB vector store wrapper for quarq RAG."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from quarq.config import QuarqConfig
from quarq.constants import (
    RAG_COLLECTION_NAME,
    RAG_HNSW_CONFIG,
    RAG_LEGACY_COLLECTION_NAMES,
    RAG_READ_BATCH_SIZE,
)
from quarq.exceptions import RAGError
from quarq.rag.loader import Document

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """A single chunk returned by a vector store query.

    Attributes:
        content: The chunk text.
        metadata: Full metadata dict (source, doc_type, date, page, chunk_id).
        similarity: Cosine similarity score in [0, 1].
        source: Shortcut to metadata['source'].
        page: Shortcut to metadata['page'].
        rerank_score: Cross-encoder relevance score, when the chunk was re-ranked.
    """

    content: str
    metadata: dict[str, str | int]
    similarity: float
    source: str
    page: int
    rerank_score: float | None = None


@dataclass
class DuplicateReport:
    """Chunk ids that repeat content already in the index."""

    exact_duplicates: list[str]
    contained_tails: list[str]


class VectorStore:
    """Persistent ChromaDB vector store for document chunks.

    Uses chunk_id as the document ID for idempotent upserts.

    Args:
        cfg: Loaded QuarqConfig. chroma_path is expanded from config.rag.chroma_path.
    """

    def __init__(self, cfg: QuarqConfig) -> None:
        import chromadb

        chroma_path = Path(cfg.rag.chroma_path).expanduser()
        chroma_path.mkdir(parents=True, exist_ok=True)

        try:
            self._client = chromadb.PersistentClient(path=str(chroma_path))
            self._collection = self._open_collection()
        except Exception as exc:
            raise RAGError(f"Failed to initialise ChromaDB at {chroma_path}: {exc}") from exc

        self._cached_count: int | None = None
        self.config_drift = self._hnsw_drift()
        if self.config_drift:
            logger.warning(
                "Collection %s has HNSW settings that differ from RAG_HNSW_CONFIG "
                "(setting: (actual, expected)): %s. ChromaDB applies them only at creation; "
                "rebuild the collection to change them.",
                RAG_COLLECTION_NAME,
                self.config_drift,
            )

    def _open_collection(self) -> Any:
        """Get the collection, creating it with RAG_HNSW_CONFIG if it doesn't exist.

        Returns:
            The ChromaDB collection.
        """
        return self._client.get_or_create_collection(
            name=RAG_COLLECTION_NAME,
            configuration=cast(Any, {"hnsw": dict(RAG_HNSW_CONFIG)}),  # chromadb TypedDict
        )

    def _hnsw_drift(self) -> dict[str, tuple[object, object]]:
        """Compare the collection's HNSW settings with RAG_HNSW_CONFIG.

        Returns:
            {setting: (actual, expected)} for every setting that differs; empty if all match.
        """
        try:
            actual = (self._collection.configuration_json or {}).get("hnsw") or {}
        except Exception as exc:  # the check must never block opening the store
            logger.warning("Could not read HNSW settings of %s: %s", RAG_COLLECTION_NAME, exc)
            return {}
        return {
            key: (actual.get(key), expected)
            for key, expected in RAG_HNSW_CONFIG.items()
            if actual.get(key) != expected
        }

    def upsert(self, documents: list[Document], embeddings: list[list[float]]) -> int:
        """Add documents to the collection, skipping existing chunk_ids.

        Args:
            documents: List of Document instances to store.
            embeddings: Parallel list of embedding vectors (one per document).

        Returns:
            Number of documents upserted (existing chunks are updated in place).

        Raises:
            RAGError: On any ChromaDB error.
        """
        if not documents:
            return 0

        seen: set[str] = set()
        deduped_docs: list[Document] = []
        deduped_embeddings: list[list[float]] = []
        for doc, emb in zip(documents, embeddings):
            cid = str(doc.metadata["chunk_id"])
            if cid not in seen:
                seen.add(cid)
                deduped_docs.append(doc)
                deduped_embeddings.append(emb)

        ids = [str(doc.metadata["chunk_id"]) for doc in deduped_docs]
        contents = [doc.content for doc in deduped_docs]
        metadatas = [
            {k: str(v) if not isinstance(v, (str, int, float, bool)) else v
             for k, v in doc.metadata.items()}
            for doc in deduped_docs
        ]

        try:
            self._collection.upsert(
                ids=ids,
                documents=contents,
                embeddings=deduped_embeddings,
                metadatas=metadatas,
            )
        except Exception as exc:
            raise RAGError(f"ChromaDB upsert failed: {exc}") from exc

        self._cached_count = None
        return len(ids)

    def query(
        self,
        embedding: list[float],
        k: int = 5,
        filters: dict | None = None,
    ) -> list[RetrievedChunk]:
        """Query the collection for the nearest neighbours to an embedding.

        Args:
            embedding: Query vector.
            k: Maximum number of results to return.
            filters: Optional ChromaDB where-clause dict (e.g. {'doc_type': 'ecb_fsr'}).

        Returns:
            List of RetrievedChunk sorted by similarity descending.

        Raises:
            RAGError: On any ChromaDB error.
        """
        try:
            if self._cached_count is None:
                self._cached_count = self._collection.count()
            n_results = min(k, max(self._cached_count, 1))
            kwargs: dict = {
                "query_embeddings": [embedding],
                "n_results": n_results,
                "include": ["documents", "metadatas", "distances"],
            }
            if filters:
                kwargs["where"] = filters

            results = self._collection.query(**kwargs)
        except Exception as exc:
            raise RAGError(f"ChromaDB query failed: {exc}") from exc

        chunks: list[RetrievedChunk] = []
        docs = results.get("documents") or [[]]
        metas = results.get("metadatas") or [[]]
        distances = results.get("distances") or [[]]

        for content, meta, distance in zip(docs[0], metas[0], distances[0]):
            # ChromaDB cosine distance: similarity = 1 - distance
            similarity = max(0.0, 1.0 - float(distance))
            chunks.append(
                RetrievedChunk(
                    content=content,
                    metadata=meta,
                    similarity=similarity,
                    source=str(meta.get("source", "")),
                    page=int(meta.get("page", 0)),
                )
            )

        return chunks

    def count(self) -> int:
        """Return total number of chunks stored in the collection.

        Returns:
            Integer chunk count.
        """
        try:
            return self._collection.count()
        except Exception as exc:
            logger.warning("VectorStore.count failed: %s", exc)
            return 0

    def reset(self) -> None:
        """Delete and recreate the collection, wiping all stored chunks.

        Raises:
            RAGError: If deletion or recreation fails.
        """
        try:
            self._client.delete_collection(RAG_COLLECTION_NAME)
            self._collection = self._open_collection()
        except Exception as exc:
            raise RAGError(f"VectorStore.reset failed: {exc}") from exc

        self._cached_count = 0

    def count_sources(self) -> int:
        """Return number of unique source documents.

        Returns:
            Integer source count.
        """
        try:
            results = self._collection.get(include=["metadatas"])
            sources = {m.get("source") for m in results["metadatas"] if m.get("source")}
            return len(sources)
        except Exception as exc:
            logger.warning("VectorStore.count_sources failed: %s", exc)
            return 0

    def replace_sources(
        self, documents: list[Document], embeddings: list[list[float]]
    ) -> tuple[int, int]:
        """Upsert documents, then remove chunks of the same sources that weren't re-produced.

        Re-indexing a file this way replaces its chunks instead of adding copies next to
        the old ones (which is how chunks stored under an older id formula were duplicated).

        Args:
            documents: Chunks to store; every source present is treated as fully re-indexed.
            embeddings: Parallel embedding vectors.

        Returns:
            (chunks upserted, stale chunks removed).

        Raises:
            RAGError: On any ChromaDB error.
        """
        added = self.upsert(documents, embeddings)
        keep: dict[str, set[str]] = {}
        for doc in documents:
            keep.setdefault(str(doc.metadata["source"]), set()).add(str(doc.metadata["chunk_id"]))
        stale: list[str] = []
        try:
            for source, new_ids in keep.items():
                existing = self._collection.get(where={"source": source}, include=[])["ids"]
                stale.extend(cid for cid in existing if cid not in new_ids)
        except Exception as exc:
            raise RAGError(f"VectorStore.replace_sources failed: {exc}") from exc
        return added, self.remove_chunks(stale)

    def find_duplicate_chunks(self, max_tail_words: int) -> DuplicateReport:
        """Find chunks that repeat content already in the index.

        Two kinds are reported:
        - exact duplicates: the same (source, page, text) stored under more than one id.
          The copy whose id matches loader.chunk_id is kept; the others are listed.
        - contained tails: short chunks (at most max_tail_words words) whose text sits
          entirely inside another chunk of the same page, as the chunker used to emit.

        Args:
            max_tail_words: Longest chunk that can count as a contained tail (the overlap).

        Returns:
            The ids to remove, by kind, each sorted.

        Raises:
            RAGError: On any ChromaDB error.
        """
        from quarq.rag.loader import chunk_id

        try:
            ids, docs, metas = [], [], []
            for offset in range(0, self._collection.count(), RAG_READ_BATCH_SIZE):
                got = self._collection.get(
                    include=["documents", "metadatas"], limit=RAG_READ_BATCH_SIZE, offset=offset
                )
                ids += got["ids"]
                docs += got["documents"]
                metas += got["metadatas"]
        except Exception as exc:
            raise RAGError(f"VectorStore.find_duplicate_chunks failed: {exc}") from exc

        groups: dict[tuple[str, int, str], list[int]] = {}
        for i, (meta, text) in enumerate(zip(metas, docs, strict=True)):
            groups.setdefault((str(meta["source"]), int(meta["page"]), text), []).append(i)

        exact: list[str] = []
        kept_by_page: dict[tuple[str, int], list[int]] = {}
        for (source, page, text), members in groups.items():
            canonical = chunk_id(source, page, text)
            keep = next(
                (i for i in members if ids[i] == canonical), min(members, key=ids.__getitem__)
            )
            exact.extend(ids[i] for i in members if i != keep)
            kept_by_page.setdefault((source, page), []).append(keep)

        tails: list[str] = []
        for members in kept_by_page.values():
            for i in members:
                if len(docs[i].split()) <= max_tail_words and any(
                    j != i and docs[i] in docs[j] and len(docs[j]) > len(docs[i]) for j in members
                ):
                    tails.append(ids[i])
        return DuplicateReport(exact_duplicates=sorted(exact), contained_tails=sorted(tails))

    def remove_chunks(self, chunk_ids: list[str]) -> int:
        """Delete chunks by id, in batches.

        Args:
            chunk_ids: Ids to delete. Ids not in the collection are ignored by ChromaDB.

        Returns:
            Number of ids passed for deletion.

        Raises:
            RAGError: On any ChromaDB error.
        """
        try:
            for start in range(0, len(chunk_ids), RAG_READ_BATCH_SIZE):
                self._collection.delete(ids=chunk_ids[start: start + RAG_READ_BATCH_SIZE])
        except Exception as exc:
            raise RAGError(f"VectorStore.remove_chunks failed: {exc}") from exc
        self._cached_count = None
        return len(chunk_ids)

    def legacy_collection_counts(self) -> dict[str, int]:
        """Return the chunk count of every legacy collection that exists in this ChromaDB.

        Returns:
            {collection name: chunk count}, only for legacy collections present.
        """
        collections = self._client.list_collections()
        existing = {c.name if hasattr(c, "name") else str(c) for c in collections}
        counts: dict[str, int] = {}
        for name in RAG_LEGACY_COLLECTION_NAMES:
            if name in existing:
                counts[name] = self._client.get_collection(name).count()
        return counts

    def migrate_from(self, legacy_name: str, batch_size: int = RAG_READ_BATCH_SIZE) -> int:
        """Copy every chunk of a legacy collection into this one, without re-embedding.

        Ids, embeddings, text and metadata are copied as they are, so running it twice
        gives the same result. The legacy collection is left untouched.

        Args:
            legacy_name: Name of the collection to copy from.
            batch_size: Chunks read and written per batch.

        Returns:
            Number of chunks copied.

        Raises:
            RAGError: If the legacy collection doesn't exist, or on any ChromaDB error.
        """
        try:
            legacy = self._client.get_collection(legacy_name)
        except Exception as exc:
            raise RAGError(f"Collection {legacy_name!r} not found: {exc}") from exc
        try:
            total = legacy.count()
            copied = 0
            for offset in range(0, total, batch_size):
                batch = legacy.get(
                    include=["embeddings", "documents", "metadatas"],
                    limit=batch_size,
                    offset=offset,
                )
                if not batch["ids"]:
                    break
                self._collection.upsert(
                    ids=batch["ids"],
                    embeddings=batch["embeddings"],
                    documents=batch["documents"],
                    metadatas=batch["metadatas"],
                )
                copied += len(batch["ids"])
        except Exception as exc:
            raise RAGError(f"Migrating {legacy_name!r} failed: {exc}") from exc
        self._cached_count = None
        return copied

    def all_records(self) -> dict[str, Any]:
        """Return every stored chunk's id, text, metadata and embedding.

        Used to build an exact-search reference for the eval. Embeddings come back as
        one float32 numpy array: as Python lists of floats they would take about ten
        times the memory (a Python float is a 24-byte object, not 4 bytes).

        Returns:
            Dict with parallel lists under "ids", "documents", "metadatas", and an
            (n_chunks, dim) float32 array under "embeddings".

        Raises:
            RAGError: On any ChromaDB error.
        """
        try:
            total = self._collection.count()
            ids: list[str] = []
            documents: list[str] = []
            metadatas: list[dict[str, Any]] = []
            matrix: np.ndarray | None = None
            # Batches keep the peak low: ChromaDB returns float64, so reading everything
            # at once briefly holds a second, double-size copy of all embeddings.
            for offset in range(0, total, RAG_READ_BATCH_SIZE):
                got = self._collection.get(
                    include=["documents", "metadatas", "embeddings"],
                    limit=RAG_READ_BATCH_SIZE,
                    offset=offset,
                )
                batch = got.get("embeddings")
                if batch is None or len(batch) == 0:
                    break
                if matrix is None:
                    matrix = np.empty((total, len(batch[0])), dtype=np.float32)
                matrix[len(ids): len(ids) + len(batch)] = batch
                ids.extend(got.get("ids") or [])
                documents.extend(got.get("documents") or [])
                metadatas.extend(got.get("metadatas") or [])
        except Exception as exc:
            raise RAGError(f"VectorStore.all_records failed: {exc}") from exc
        return {
            "ids": ids,
            "documents": documents,
            "metadatas": metadatas,
            "embeddings": (
                np.zeros((0, 0), dtype=np.float32) if matrix is None else matrix[: len(ids)]
            ),
        }

    def list_sources(self) -> set[str]:
        """Return the set of source filenames in the collection.

        Returns:
            Source filenames.

        Raises:
            RAGError: On any ChromaDB error.
        """
        try:
            results = self._collection.get(include=["metadatas"])
        except Exception as exc:
            raise RAGError(f"VectorStore.list_sources failed: {exc}") from exc
        return {str(m["source"]) for m in results.get("metadatas") or [] if m.get("source")}

    def update_source_metadata(
        self, source: str, fields: dict[str, str | int | float | bool]
    ) -> int:
        """Merge fields into the metadata of every chunk from one source, without re-embedding.

        Each chunk's full metadata is read, merged in Python, and written back whole,
        so existing keys are kept whatever ChromaDB's own merge behaviour is.

        Args:
            source: Source filename whose chunks to update.
            fields: Keys to add or overwrite (ChromaDB scalar values only).

        Returns:
            Number of chunks updated.

        Raises:
            RAGError: On any ChromaDB error.
        """
        try:
            results = self._collection.get(where={"source": source}, include=["metadatas"])
            ids = results.get("ids") or []
            if not ids:
                return 0
            merged = [{**meta, **fields} for meta in results["metadatas"]]
            self._collection.update(ids=ids, metadatas=merged)
        except Exception as exc:
            raise RAGError(
                f"VectorStore.update_source_metadata failed for {source}: {exc}"
            ) from exc
        return len(ids)

    def list_chunks(self) -> list[RetrievedChunk]:
        """Return every stored chunk with its metadata.

        Used by the eval draft generator to sample passages. similarity is 0.0
        because no query was run.

        Returns:
            All chunks in the collection.

        Raises:
            RAGError: On any ChromaDB error.
        """
        try:
            results = self._collection.get(include=["documents", "metadatas"])
        except Exception as exc:
            raise RAGError(f"VectorStore.list_chunks failed: {exc}") from exc

        docs = results.get("documents") or []
        metas = results.get("metadatas") or []
        return [
            RetrievedChunk(
                content=content,
                metadata=meta,
                similarity=0.0,
                source=str(meta.get("source", "")),
                page=int(meta.get("page", 0)),
            )
            for content, meta in zip(docs, metas, strict=False)
        ]
