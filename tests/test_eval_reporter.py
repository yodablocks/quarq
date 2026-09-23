"""Tests for quarq.eval.reporter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from rich.console import Console

from quarq.eval.reporter import render_table, report_paths, to_json, to_markdown
from quarq.eval.runner import EvalResult, PerQuestionResult
from quarq.exceptions import EvalError


def _result() -> EvalResult:
    good = PerQuestionResult(
        id="q1", question="Good question?", doc_type="ecb_fsr", gold=[("a.pdf", 1)],
        retrieved=[("a.pdf", 1)], n_above_threshold=1,
        metrics={"precision@1": 1.0, "recall@1": 1.0, "hit@1": 1.0, "mrr": 1.0},
    )
    bad = PerQuestionResult(
        id="q2", question="Bad | question?", doc_type="amf_sfdr", gold=[("d.pdf", 4)],
        retrieved=[("b.pdf", 2)], n_above_threshold=1,
        metrics={"precision@1": 0.0, "recall@1": 0.0, "hit@1": 0.0, "mrr": 0.0},
    )
    return EvalResult(
        dataset_name="quarq_gold_v1", n_questions=2, k_values=[1], use_doc_type_filter=False,
        provenance_counts={"hand-written": 2},
        config_snapshot={"min_similarity": 0.35, "chunk_size": 512},
        aggregate={"precision@1": 0.5, "recall@1": 0.5, "hit@1": 0.5, "mrr": 0.5},
        per_question=[good, bad], corpus_chunk_count=3333,
        generated_at="2026-09-23T16:41:05+00:00",
    )


def test_report_paths_are_timestamped(tmp_path: Path) -> None:
    json_path, md_path = report_paths(_result(), tmp_path)
    assert json_path == tmp_path / "eval_report_20260923T164105.json"
    assert md_path == tmp_path / "eval_report_20260923T164105.md"


def test_to_json_round_trips(tmp_path: Path) -> None:
    path = to_json(_result(), tmp_path / "r.json")

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["aggregate"]["recall@1"] == 0.5
    assert data["config_snapshot"]["min_similarity"] == 0.35
    assert data["per_question"][1]["retrieved"] == [["b.pdf", 2]]


def test_writers_refuse_to_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "r.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(EvalError, match="already exists"):
        to_json(_result(), path)
    with pytest.raises(EvalError, match="already exists"):
        to_markdown(_result(), path)


def test_to_markdown_has_config_results_and_worst_questions(tmp_path: Path) -> None:
    text = to_markdown(_result(), tmp_path / "r.md").read_text(encoding="utf-8")

    assert "# quarq retrieval eval: quarq_gold_v1" in text
    assert "| min_similarity | 0.35 |" in text
    assert "Provenance: hand-written (2)" in text
    assert "| 1 | 0.500 | 0.500 | 0.500 |" in text
    assert "MRR: 0.500" in text
    worst = text.split("## Worst questions")[1]
    assert worst.index("q2") < worst.index("q1")
    assert "Bad \\| question?" in worst  # pipes escaped so the table survives


def test_to_json_wraps_oserror_from_out_path_component_being_a_file(tmp_path: Path) -> None:
    (tmp_path / "afile").write_text("x", encoding="utf-8")
    with pytest.raises(EvalError, match="Cannot write report"):
        to_json(_result(), tmp_path / "afile" / "r.json")


def test_render_table_mentions_every_k_and_mrr() -> None:
    console = Console(record=True, width=120)
    console.print(render_table(_result()))
    out = console.export_text()
    assert "Precision@k" in out and "Recall@k" in out and "Hit@k" in out
    assert "MRR" in out
    assert "0.500" in out
