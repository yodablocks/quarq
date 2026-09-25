"""ChromaDB vector store wrapper for quarq RAG."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from quarq.config import QuarqConfig
from quarq.constants import (
    RAG_COLLECTION_NAME,
    RAG_HNSW_CONFIG,
    RAG_LEGACY_COLLECTION_NAMES,
    RAG_MIGRATE_BATCH_SIZE,
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
    """

    content: str
    metadata: dict[str, str | int]
    similarity: float
    source: str
    page: int


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

    def migrate_from(self, legacy_name: str, batch_size: int = RAG_MIGRATE_BATCH_SIZE) -> int:
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

    def all_records(self) -> dict[str, list]:
        """Return every stored chunk's id, text, metadata and embedding.

        Used to build an exact-search reference for the eval.

        Returns:
            Dict with parallel lists under "ids", "documents", "metadatas", "embeddings".

        Raises:
            RAGError: On any ChromaDB error.
        """
        try:
            got = self._collection.get(include=["documents", "metadatas", "embeddings"])
        except Exception as exc:
            raise RAGError(f"VectorStore.all_records failed: {exc}") from exc
        embeddings = got.get("embeddings")
        return {
            "ids": list(got.get("ids") or []),
            "documents": list(got.get("documents") or []),
            "metadatas": list(got.get("metadatas") or []),
            "embeddings": [] if embeddings is None else [list(e) for e in embeddings],
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
