"""ChromaDB vector store wrapper for quarq RAG."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from quarq.config import QuarqConfig
from quarq.constants import RAG_COLLECTION_NAME
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
            self._collection = self._client.get_or_create_collection(
                name=RAG_COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as exc:
            raise RAGError(f"Failed to initialise ChromaDB at {chroma_path}: {exc}") from exc

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
            kwargs: dict = {
                "query_embeddings": [embedding],
                "n_results": min(k, max(self.count(), 1)),
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
