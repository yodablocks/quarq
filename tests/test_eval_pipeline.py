"""Tests for quarq.eval.pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from quarq.config import QuarqConfig
from quarq.eval.pipeline import evaluate
from quarq.exceptions import EvalError

ROW = {
    "id": "q1",
    "question": "Q one?",
    "gold": [{"source": "a.pdf", "page": 1}],
    "doc_type": "ecb_fsr",
    "provenance": "hand-written",
}


def test_evaluate_writes_both_reports(tmp_path: Path, fake_retriever_cls) -> None:
    dataset = tmp_path / "gold_x.jsonl"
    dataset.write_text(json.dumps(ROW) + "\n", encoding="utf-8")
    retriever = fake_retriever_cls({"Q one?": [("a.pdf", 1)]})

    result, json_path, md_path = evaluate(
        retriever, QuarqConfig(), dataset_path=dataset, out_dir=tmp_path / "reports",
        corpus_chunk_count=7,
    )

    assert result.dataset_name == "gold_x"
    assert result.aggregate["recall@5"] == 1.0
    assert json_path.exists() and md_path.exists()
    assert json_path.parent == tmp_path / "reports"


def test_evaluate_rejects_gold_refs_missing_from_corpus(
    tmp_path: Path, fake_retriever_cls
) -> None:
    dataset = tmp_path / "gold_x.jsonl"
    dataset.write_text(json.dumps(ROW) + "\n", encoding="utf-8")
    retriever = fake_retriever_cls({"Q one?": [("a.pdf", 1)]})

    with pytest.raises(EvalError, match="q1: a.pdf p1") as exc_info:
        evaluate(
            retriever, QuarqConfig(), dataset_path=dataset, out_dir=tmp_path / "reports",
            corpus_chunk_count=7, known_refs={("other.pdf", 1)},
        )
    assert "page index" in str(exc_info.value)
    assert not (tmp_path / "reports").exists()


def test_evaluate_succeeds_when_gold_refs_are_known(tmp_path: Path, fake_retriever_cls) -> None:
    dataset = tmp_path / "gold_x.jsonl"
    dataset.write_text(json.dumps(ROW) + "\n", encoding="utf-8")
    retriever = fake_retriever_cls({"Q one?": [("a.pdf", 1)]})

    result, json_path, md_path = evaluate(
        retriever, QuarqConfig(), dataset_path=dataset, out_dir=tmp_path / "reports",
        corpus_chunk_count=7, known_refs={("a.pdf", 1)},
    )

    assert result.dataset_name == "gold_x"
    assert json_path.exists() and md_path.exists()


def test_evaluate_refuses_unaccepted_drafts_end_to_end(
    tmp_path: Path, fake_retriever_cls
) -> None:
    draft_row = {**ROW, "provenance": "synthetic-draft"}
    dataset = tmp_path / "gold_x.jsonl"
    dataset.write_text(json.dumps(draft_row) + "\n", encoding="utf-8")
    retriever = fake_retriever_cls({"Q one?": [("a.pdf", 1)]})
    out_dir = tmp_path / "reports"

    with pytest.raises(EvalError, match="not human-accepted"):
        evaluate(
            retriever, QuarqConfig(), dataset_path=dataset, out_dir=out_dir,
            corpus_chunk_count=7,
        )

    assert not out_dir.exists()
