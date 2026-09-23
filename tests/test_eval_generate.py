"""Tests for quarq.eval.generate and `quarq eval-gen` (fake LLM, fake store)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

import quarq.cli as cli
from quarq.config import QuarqConfig
from quarq.eval.dataset import DRAFT_PROVENANCE, load_gold
from quarq.eval.generate import clean_question, draft_items, sample_chunks
from quarq.rag.store import RetrievedChunk

LONG = "x" * 250


def _chunk(source: str, page: int, doc_type: str, content: str = LONG) -> RetrievedChunk:
    return RetrievedChunk(
        content=content,
        metadata={"source": source, "page": page, "doc_type": doc_type,
                  "chunk_id": f"{source}-{page}-cid"},
        similarity=0.0,
        source=source,
        page=page,
    )


class _LLM:
    def __init__(self, replies: list[str]) -> None:
        self.replies = replies
        self.prompts: list[str] = []

    def generate(self, prompt: str, system: str = "") -> str:
        self.prompts.append(prompt)
        return self.replies[len(self.prompts) - 1]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("What did the ECB flag?", "What did the ECB flag?"),
        ('  "What did the ECB flag?"  \nextra line', "What did the ECB flag?"),
        ("1. What did the ECB flag?", "What did the ECB flag?"),
        ("Question: What did the ECB flag?", "What did the ECB flag?"),
        ("This is not a question.", None),
        ("   \n  ", None),
    ],
)
def test_clean_question(raw: str, expected: str | None) -> None:
    assert clean_question(raw) == expected


def test_sample_chunks_is_balanced_deterministic_and_skips_short() -> None:
    chunks = [_chunk("e.pdf", p, "ecb_fsr") for p in range(1, 11)]
    chunks += [_chunk("b.pdf", p, "bdf_fsr") for p in range(1, 4)]
    chunks.append(_chunk("b.pdf", 99, "bdf_fsr", content="short"))

    first = sample_chunks(chunks, per_doc_type=5, seed=7)
    second = sample_chunks(list(reversed(chunks)), per_doc_type=5, seed=7)

    assert [c.page for c in first] == [c.page for c in second]
    by_type = [c.metadata["doc_type"] for c in first]
    assert by_type.count("ecb_fsr") == 5
    assert by_type.count("bdf_fsr") == 3  # only 3 eligible
    assert all(c.page != 99 for c in first)


def test_draft_items_labels_chunk_page_and_skips_non_questions() -> None:
    chunks = [_chunk("e.pdf", 4, "ecb_fsr"), _chunk("a.pdf", 9, "amf_sfdr")]
    llm = _LLM(["What rose in 2025?", "no question here"])

    items, skipped = draft_items(chunks, llm)

    assert skipped == 1
    assert len(items) == 1
    item = items[0]
    assert item.id == "draft001"
    assert item.question == "What rose in 2025?"
    assert item.gold_keys() == {("e.pdf", 4)}
    assert item.doc_type == "ecb_fsr"
    assert item.provenance == DRAFT_PROVENANCE
    assert LONG in llm.prompts[0]


class _Store:
    def __init__(self, cfg: QuarqConfig) -> None:
        pass

    def count(self) -> int:
        return 2

    def list_chunks(self) -> list[RetrievedChunk]:
        return [_chunk("e.pdf", 4, "ecb_fsr"), _chunk("a.pdf", 9, "amf_sfdr")]


def test_cmd_eval_gen_writes_drafts_and_refuses_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "load_config", lambda: QuarqConfig())
    monkeypatch.setattr("quarq.rag.store.VectorStore", _Store)
    monkeypatch.setattr("quarq.llm.get_llm", lambda cfg, agent="research": _LLM(["A?", "B?"]))
    out = tmp_path / "drafts.jsonl"

    assert cli._cmd_eval_gen(5, 7, str(out)) == 0
    drafts = load_gold(out, require_accepted=False)
    assert {d.question for d in drafts} == {"A?", "B?"}

    assert cli._cmd_eval_gen(5, 7, str(out)) == 1  # existing file is never overwritten


def test_cmd_eval_gen_rejects_non_positive_per_doc_type(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "load_config", lambda: QuarqConfig())
    monkeypatch.setattr("quarq.rag.store.VectorStore", _Store)
    monkeypatch.setattr("quarq.llm.get_llm", lambda cfg, agent="research": _LLM(["A?", "B?"]))
    out = tmp_path / "drafts.jsonl"

    assert cli._cmd_eval_gen(0, 7, str(out)) == 1
    assert cli._cmd_eval_gen(-1, 7, str(out)) == 1
    assert not out.exists()


def test_main_dispatches_eval_gen_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    def fake(per_doc_type: int, seed: int, out: str) -> int:
        seen.update(per_doc_type=per_doc_type, seed=seed, out=out)
        return 0

    monkeypatch.setattr(cli, "_cmd_eval_gen", fake)
    monkeypatch.setattr(sys, "argv", ["quarq", "eval-gen", "--per-doc-type", "3"])

    cli.main()

    assert seen == {"per_doc_type": 3, "seed": 7, "out": "reports/eval_gen_drafts.jsonl"}
