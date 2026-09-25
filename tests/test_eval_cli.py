"""Tests for the `quarq eval` CLI command (no ChromaDB, no model download)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import quarq.cli as cli
from quarq.config import QuarqConfig
from quarq.eval.index_check import IndexCheck
from quarq.rag.store import RetrievedChunk

ROW = {
    "id": "q1",
    "question": "Q one?",
    "gold": [{"source": "a.pdf", "page": 1}],
    "doc_type": "ecb_fsr",
    "provenance": "hand-written",
}


class _Store:
    chunks = 10

    def __init__(self, cfg: QuarqConfig) -> None:
        pass

    legacy: dict[str, int] = {}

    def count(self) -> int:
        return self.chunks

    def legacy_collection_counts(self) -> dict[str, int]:
        return self.legacy

    def list_chunks(self) -> list[RetrievedChunk]:
        return [
            RetrievedChunk(
                content="chunk from a.pdf p1",
                metadata={"source": "a.pdf", "page": 1},
                similarity=0.0,
                source="a.pdf",
                page=1,
            )
        ]


class _Embedder:
    def __init__(self, model_name: str | None = None) -> None:
        pass


@pytest.fixture
def patched(monkeypatch: pytest.MonkeyPatch, fake_retriever_cls) -> None:
    monkeypatch.setattr(cli, "load_config", lambda: QuarqConfig())
    monkeypatch.setattr("quarq.rag.store.VectorStore", _Store)
    monkeypatch.setattr("quarq.rag.embedder.Embedder", _Embedder)
    # The fake store can't do exact search; stub the index check with a fixed result.
    monkeypatch.setattr("quarq.rag.exact.ExactStore", lambda store: store)
    monkeypatch.setattr(
        "quarq.eval.index_check.check_index",
        lambda gold, *args, **kwargs: IndexCheck(
            n_questions=len(gold), candidates=20, questions_with_gap=0,
            missed_chunks=0, exact_match_questions=len(gold),
        ),
    )
    monkeypatch.setattr(
        "quarq.rag.retriever.build_retriever",
        lambda store, embedder, cfg: fake_retriever_cls({"Q one?": [("a.pdf", 1)]}),
    )


def test_cmd_eval_happy_path_writes_reports(tmp_path: Path, patched) -> None:
    dataset = tmp_path / "gold.jsonl"
    dataset.write_text(json.dumps(ROW) + "\n", encoding="utf-8")
    out = tmp_path / "reports"

    code = cli._cmd_eval(str(dataset), "1,3,5", str(out), False)

    assert code == 0
    assert len(list(out.glob("eval_report_*.json"))) == 1
    assert len(list(out.glob("eval_report_*.md"))) == 1
    assert "## Index check" in next(out.glob("eval_report_*.md")).read_text()


def test_cmd_eval_empty_corpus_returns_1(
    tmp_path: Path, patched, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(_Store, "chunks", 0)

    assert cli._cmd_eval(str(tmp_path / "gold.jsonl"), "1,3,5", str(tmp_path), False) == 1


def test_cmd_eval_empty_corpus_suggests_migrate_when_legacy_exists(
    tmp_path: Path, patched, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(_Store, "chunks", 0)
    monkeypatch.setattr(_Store, "legacy", {"quarq_rag_v1": 4235})

    assert cli._cmd_eval(str(tmp_path / "gold.jsonl"), "1,3,5", str(tmp_path), False) == 1
    out = " ".join(capsys.readouterr().out.split())  # undo console line wrapping
    assert "quarq rag migrate" in out
    assert "4235" in out


def test_cmd_eval_bad_dataset_returns_1(tmp_path: Path, patched) -> None:
    assert cli._cmd_eval(str(tmp_path / "missing.jsonl"), "1,3,5", str(tmp_path), False) == 1


def test_cmd_eval_rejects_gold_ref_missing_from_corpus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_retriever_cls
) -> None:
    monkeypatch.setattr(cli, "load_config", lambda: QuarqConfig())
    monkeypatch.setattr("quarq.rag.store.VectorStore", _Store)
    monkeypatch.setattr("quarq.rag.embedder.Embedder", _Embedder)
    monkeypatch.setattr(
        "quarq.rag.retriever.build_retriever",
        lambda store, embedder, cfg: fake_retriever_cls({"Q one?": [("a.pdf", 1)]}),
    )
    other_row = {
        "id": "q1",
        "question": "Q one?",
        "gold": [{"source": "other.pdf", "page": 9}],
        "doc_type": "ecb_fsr",
        "provenance": "hand-written",
    }
    dataset = tmp_path / "gold.jsonl"
    dataset.write_text(json.dumps(other_row) + "\n", encoding="utf-8")
    out = tmp_path / "reports"

    code = cli._cmd_eval(str(dataset), "1,3,5", str(out), False)

    assert code == 1
    assert not out.exists()


def test_main_dispatches_eval_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    def fake_cmd_eval(
        dataset: str | None, k: str, out: str, doc_type_filter: bool, no_rerank: bool = False
    ) -> int:
        seen.update(dataset=dataset, k=k, out=out, doc_type_filter=doc_type_filter,
                    no_rerank=no_rerank)
        return 0

    monkeypatch.setattr(cli, "_cmd_eval", fake_cmd_eval)
    monkeypatch.setattr(
        sys, "argv", ["quarq", "eval", "--dataset", "g.jsonl", "--k", "1,10", "--doc-type-filter"]
    )

    cli.main()

    assert seen == {"dataset": "g.jsonl", "k": "1,10", "out": "reports", "doc_type_filter": True,
                    "no_rerank": False}
