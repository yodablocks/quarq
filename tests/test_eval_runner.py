"""Tests for quarq.eval.runner using a scripted fake retriever (no ChromaDB)."""

from __future__ import annotations

import pytest

from quarq.config import QuarqConfig
from quarq.eval.dataset import GoldItem, GoldRef
from quarq.eval.runner import parse_k_values, run_eval
from quarq.exceptions import EvalError


def _gold() -> list[GoldItem]:
    return [
        GoldItem(id="q1", question="Q one?", gold=[GoldRef(source="a.pdf", page=1)],
                 doc_type="ecb_fsr", provenance="hand-written"),
        GoldItem(id="q2", question="Q two?", gold=[GoldRef(source="d.pdf", page=4)],
                 doc_type="amf_sfdr", provenance="hand-written"),
    ]


def test_run_eval_aggregates_hand_computed_means(fake_retriever_cls) -> None:
    retriever = fake_retriever_cls(
        {"Q one?": [("b.pdf", 2), ("a.pdf", 1), ("c.pdf", 3)], "Q two?": []}
    )

    result = run_eval(
        retriever, _gold(), QuarqConfig(), dataset_name="t", corpus_chunk_count=10,
        k_values=(1, 3),
    )

    assert result.n_questions == 2
    assert result.aggregate["precision@1"] == 0.0
    assert result.aggregate["precision@3"] == pytest.approx((1 / 3 + 0.0) / 2)
    assert result.aggregate["recall@3"] == 0.5
    assert result.aggregate["hit@1"] == 0.0
    assert result.aggregate["hit@3"] == 0.5
    assert result.aggregate["mrr"] == 0.25
    assert list(result.aggregate) == [
        "precision@1", "recall@1", "hit@1", "precision@3", "recall@3", "hit@3", "mrr",
    ]


def test_run_eval_retrieves_once_per_question_at_max_k_unfiltered(fake_retriever_cls) -> None:
    retriever = fake_retriever_cls({})

    run_eval(retriever, _gold(), QuarqConfig(), dataset_name="t", corpus_chunk_count=1,
             k_values=(1, 3, 5))

    assert [c["k"] for c in retriever.calls] == [5, 5]
    assert [c["doc_type"] for c in retriever.calls] == [None, None]
    # min_similarity is left to the retriever's config default (production behaviour).
    assert [c["min_similarity"] for c in retriever.calls] == [None, None]


def test_run_eval_doc_type_filter_passes_item_doc_type(fake_retriever_cls) -> None:
    retriever = fake_retriever_cls({})

    result = run_eval(retriever, _gold(), QuarqConfig(), dataset_name="t",
                      corpus_chunk_count=1, use_doc_type_filter=True)

    assert [c["doc_type"] for c in retriever.calls] == ["ecb_fsr", "amf_sfdr"]
    assert result.use_doc_type_filter is True


def test_run_eval_records_per_question_detail_and_snapshot(fake_retriever_cls) -> None:
    retriever = fake_retriever_cls({"Q one?": [("a.pdf", 1), ("a.pdf", 1)]})

    result = run_eval(retriever, _gold(), QuarqConfig(), dataset_name="gold_v1",
                      corpus_chunk_count=3333)

    first = result.per_question[0]
    assert first.id == "q1"
    assert first.retrieved == [("a.pdf", 1), ("a.pdf", 1)]
    assert first.n_above_threshold == 2
    assert first.metrics["mrr"] == 1.0
    assert result.per_question[1].n_above_threshold == 0
    assert result.config_snapshot["min_similarity"] == 0.35
    assert result.config_snapshot["collection"] == "quarq_rag_v1"
    assert result.config_snapshot["embedder_model"] == "intfloat/multilingual-e5-large"
    assert result.corpus_chunk_count == 3333
    assert result.dataset_name == "gold_v1"
    assert result.generated_at.endswith("+00:00")
    assert result.provenance_counts == {"hand-written": 2}


def test_parse_k_values_sorts_dedupes_and_validates() -> None:
    assert parse_k_values("5,1,3,3") == (1, 3, 5)
    with pytest.raises(EvalError, match="positive integers"):
        parse_k_values("0,3")
    with pytest.raises(EvalError, match="positive integers"):
        parse_k_values("a,b")
