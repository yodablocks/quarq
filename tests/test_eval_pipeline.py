"""Tests for quarq.eval.pipeline."""

from __future__ import annotations

import json
from pathlib import Path

from quarq.config import QuarqConfig
from quarq.eval.pipeline import evaluate

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
