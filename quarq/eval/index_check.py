"""Check how far the vector index is from exact search, as part of `quarq eval`.

Two numbers, measured on the gold questions:

- Raw index recall: when asked for the candidates the retriever needs
  (max k x RETRIEVAL_OVERFETCH_FACTOR), how many true nearest neighbours does the
  index leave out? Ties between near-duplicate chunks are not counted.
- End-to-end exactness: for how many questions are quarq's final retrieved pages
  (after the similarity floor, doc_type filter and per-page dedupe) the same as the
  identical pipeline run on exact search?
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, cast

from quarq.config import QuarqConfig
from quarq.constants import RETRIEVAL_OVERFETCH_FACTOR, SIMILARITY_TIE_TOLERANCE
from quarq.eval.dataset import GoldItem
from quarq.rag.retriever import Retriever
from quarq.rag.store import RetrievedChunk


class StoreLike(Protocol):
    """The subset of VectorStore the index check and Retriever use."""

    def query(
        self, embedding: list[float], k: int = 5, filters: dict | None = None
    ) -> list[RetrievedChunk]: ...

    def count(self) -> int: ...


class EmbedderLike(Protocol):
    """The subset of Embedder the index check uses."""

    def embed_query(self, query: str) -> list[float]: ...


@dataclass
class IndexCheck:
    """How the vector index compares with exact search on the gold questions."""

    n_questions: int
    candidates: int
    questions_with_gap: int
    missed_chunks: int
    exact_match_questions: int


def missed_neighbours(index: list[RetrievedChunk], exact: list[RetrievedChunk]) -> int:
    """Count exact-search chunks the index left out despite scoring higher than its worst result.

    Args:
        index: Chunks the index returned for a query.
        exact: Chunks exact search returned for the same query and size.

    Returns:
        Number of true neighbours missing from the index result, ties excluded.
    """
    returned = {c.metadata.get("chunk_id") for c in index}
    floor = min((c.similarity for c in index), default=float("-inf"))
    return sum(
        1
        for c in exact
        if c.metadata.get("chunk_id") not in returned
        and c.similarity > floor + SIMILARITY_TIE_TOLERANCE
    )


def check_index(
    gold: list[GoldItem],
    index_store: StoreLike,
    exact_store: StoreLike,
    embedder: EmbedderLike,
    cfg: QuarqConfig,
    *,
    max_k: int,
    use_doc_type_filter: bool,
) -> IndexCheck:
    """Compare the index with exact search on every gold question.

    Args:
        gold: Gold items whose questions are used as queries.
        index_store: The production store (ChromaDB HNSW).
        exact_store: Brute-force reference over the same chunks.
        embedder: Embeds the questions.
        cfg: Loaded config (retrieval defaults).
        max_k: Largest k the eval reports.
        use_doc_type_filter: Restrict each query to the item's doc_type, as the eval does.

    Returns:
        The IndexCheck summary.
    """
    candidates = max_k * RETRIEVAL_OVERFETCH_FACTOR
    index_retriever = Retriever(cast(Any, index_store), cast(Any, embedder), cfg)
    exact_retriever = Retriever(cast(Any, exact_store), cast(Any, embedder), cfg)
    with_gap = missed_total = exact_matches = 0
    for item in gold:
        doc_type = item.doc_type if use_doc_type_filter else None
        filters = {"doc_type": doc_type} if doc_type else None
        embedding = embedder.embed_query(item.question)
        missed = missed_neighbours(
            index_store.query(embedding, k=candidates, filters=filters),
            exact_store.query(embedding, k=candidates, filters=filters),
        )
        with_gap += missed > 0
        missed_total += missed

        pages = [(c.source, c.page) for c in index_retriever.retrieve(item.question, k=max_k,
                                                                      doc_type=doc_type)]
        exact_pages = [(c.source, c.page) for c in exact_retriever.retrieve(item.question,
                                                                            k=max_k,
                                                                            doc_type=doc_type)]
        exact_matches += pages == exact_pages
    return IndexCheck(
        n_questions=len(gold),
        candidates=candidates,
        questions_with_gap=with_gap,
        missed_chunks=missed_total,
        exact_match_questions=exact_matches,
    )
