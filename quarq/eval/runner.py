"""Run a gold set through a retriever and compute retrieval metrics."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

from quarq.config import QuarqConfig
from quarq.constants import DEFAULT_K_VALUES, RAG_COLLECTION_NAME
from quarq.eval.dataset import GoldItem, Ref
from quarq.eval.index_check import IndexCheck
from quarq.eval.metrics import hit_at_k, mean, precision_at_k, recall_at_k, reciprocal_rank
from quarq.exceptions import EvalError
from quarq.rag.store import RetrievedChunk


class RetrieverLike(Protocol):
    """The subset of quarq.rag.retriever.Retriever the runner depends on."""

    def retrieve(
        self,
        query: str,
        k: int | None = None,
        doc_type: str | None = None,
        min_similarity: float | None = None,
    ) -> list[RetrievedChunk]: ...


@dataclass
class PerQuestionResult:
    """Retrieval outcome and metrics for one gold question."""

    id: str
    question: str
    doc_type: str | None
    gold: list[Ref]
    retrieved: list[Ref]
    n_above_threshold: int
    metrics: dict[str, float]


@dataclass
class EvalResult:
    """Everything one eval run produced, including the settings that made it."""

    dataset_name: str
    n_questions: int
    k_values: list[int]
    use_doc_type_filter: bool
    provenance_counts: dict[str, int]
    config_snapshot: dict[str, object]
    aggregate: dict[str, float]
    per_question: list[PerQuestionResult]
    corpus_chunk_count: int
    generated_at: str
    index_check: IndexCheck | None = None


def parse_k_values(raw: str) -> tuple[int, ...]:
    """Parse a comma-separated list of k values.

    Args:
        raw: Text such as "1,3,5".

    Returns:
        Sorted, de-duplicated k values.

    Raises:
        EvalError: If any value is not a positive integer.
    """
    try:
        values = {int(part) for part in raw.split(",") if part.strip()}
    except ValueError as exc:
        raise EvalError(f"k values must be positive integers, got {raw!r}") from exc
    if not values or min(values) < 1:
        raise EvalError(f"k values must be positive integers, got {raw!r}")
    return tuple(sorted(values))


def config_snapshot(cfg: QuarqConfig) -> dict[str, object]:
    """Capture the retrieval settings that determine eval results.

    Args:
        cfg: Loaded quarq configuration.

    Returns:
        Settings dict stored with every result so runs are comparable.
    """
    return {
        "chunk_size": cfg.rag.chunk_size,
        "chunk_overlap": cfg.rag.chunk_overlap,
        "top_k": cfg.rag.top_k,
        "min_similarity": cfg.rag.min_similarity,
        "embedder_model": cfg.embedder.model,
        "rerank": cfg.rag.rerank,
        "reranker_model": cfg.rag.reranker_model if cfg.rag.rerank else None,
        "rerank_top_n": cfg.rag.rerank_top_n if cfg.rag.rerank else None,
        "collection": RAG_COLLECTION_NAME,
    }


def _question_metrics(
    retrieved: list[Ref], gold: set[Ref], k_values: tuple[int, ...]
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for k in k_values:
        metrics[f"precision@{k}"] = precision_at_k(retrieved, gold, k)
        metrics[f"recall@{k}"] = recall_at_k(retrieved, gold, k)
        metrics[f"hit@{k}"] = hit_at_k(retrieved, gold, k)
    metrics["mrr"] = reciprocal_rank(retrieved, gold)
    return metrics


def run_eval(
    retriever: RetrieverLike,
    gold: list[GoldItem],
    cfg: QuarqConfig,
    *,
    dataset_name: str,
    corpus_chunk_count: int,
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
    use_doc_type_filter: bool = False,
) -> EvalResult:
    """Retrieve for every gold question and score the results.

    Retrieval runs once per question at max(k_values) and is sliced per k.
    min_similarity is not passed, so the retriever applies the production
    threshold from config; chunks it drops count as misses.

    Args:
        retriever: Object with Retriever.retrieve's signature.
        gold: Accepted gold items.
        cfg: Loaded config, snapshotted into the result.
        dataset_name: Label stored in the result (usually the file stem).
        corpus_chunk_count: Chunks in the collection at run time.
        k_values: Cut-offs to report.
        use_doc_type_filter: Restrict each retrieval to the item's doc_type.

    Returns:
        The full EvalResult.

    Raises:
        EvalError: If gold or k_values is empty.
        RAGError: Propagated from the retriever.
    """
    if not gold:
        raise EvalError("gold set is empty")
    if not k_values:
        raise EvalError("k_values is empty")
    max_k = max(k_values)

    per_question: list[PerQuestionResult] = []
    for item in gold:
        chunks = retriever.retrieve(
            item.question,
            k=max_k,
            doc_type=item.doc_type if use_doc_type_filter else None,
        )
        retrieved: list[Ref] = [(c.source, int(c.page)) for c in chunks]
        per_question.append(
            PerQuestionResult(
                id=item.id,
                question=item.question,
                doc_type=item.doc_type,
                gold=sorted(item.gold_keys()),
                retrieved=retrieved,
                n_above_threshold=len(chunks),
                metrics=_question_metrics(retrieved, item.gold_keys(), k_values),
            )
        )

    keys = list(per_question[0].metrics)
    aggregate = {key: mean([q.metrics[key] for q in per_question]) for key in keys}
    provenance_counts = dict(sorted(Counter(item.provenance for item in gold).items()))

    return EvalResult(
        dataset_name=dataset_name,
        n_questions=len(per_question),
        k_values=list(k_values),
        use_doc_type_filter=use_doc_type_filter,
        provenance_counts=provenance_counts,
        config_snapshot=config_snapshot(cfg),
        aggregate=aggregate,
        per_question=per_question,
        corpus_chunk_count=corpus_chunk_count,
        generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )
