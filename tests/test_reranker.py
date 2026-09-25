"""Tests for the cross-encoder re-ranker and its use in the retriever (model mocked)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from quarq.rag.store import RetrievedChunk


def _chunk(text: str, sim: float, source: str = "a.pdf", page: int = 1) -> RetrievedChunk:
    return RetrievedChunk(content=text, metadata={"source": source, "page": page},
                          similarity=sim, source=source, page=page)


class _FakeCrossEncoder:
    """Scores a pair by a number written in the passage text, e.g. 'score=0.7'."""

    loads = 0

    def __init__(self, model_name: str, max_length: int, device: str | None = None) -> None:
        type(self).loads += 1
        self.model_name, self.max_length, self.device = model_name, max_length, device

    def predict(self, pairs, **kwargs):
        return [float(text.split("score=")[1].split()[0]) for _, text in pairs]


@pytest.fixture
def fake_ce(monkeypatch: pytest.MonkeyPatch):
    _FakeCrossEncoder.loads = 0
    monkeypatch.setattr("sentence_transformers.CrossEncoder", _FakeCrossEncoder)
    return _FakeCrossEncoder


def test_config_defaults_enable_reranking() -> None:
    from quarq.config import QuarqConfig

    rag = QuarqConfig().rag
    assert rag.rerank is True
    assert rag.reranker_model  # set in config, never hardcoded at the call site
    assert rag.rerank_top_n == 10
    assert rag.rerank_max_length == 512


def test_reranker_orders_by_score_and_records_it(fake_ce) -> None:
    from quarq.rag.reranker import Reranker

    reranker = Reranker("some/model", max_length=256)
    out = reranker.rerank("q", [_chunk("score=0.1 a", 0.9), _chunk("score=0.8 b", 0.5)])

    assert [c.content for c in out] == ["score=0.8 b", "score=0.1 a"]
    assert out[0].rerank_score == pytest.approx(0.8)


def test_reranker_loads_the_model_once_and_lazily(fake_ce) -> None:
    from quarq.rag.reranker import Reranker

    reranker = Reranker("some/model", max_length=256)
    assert fake_ce.loads == 0  # constructing it does not load 2 GB of weights
    reranker.rerank("q", [_chunk("score=0.1", 0.9)])
    reranker.rerank("q", [_chunk("score=0.2", 0.9)])
    assert fake_ce.loads == 1


def test_reranker_with_no_candidates_skips_the_model(fake_ce) -> None:
    from quarq.rag.reranker import Reranker

    assert Reranker("some/model", max_length=256).rerank("q", []) == []
    assert fake_ce.loads == 0


def test_retriever_reranks_only_the_top_n_then_dedupes_pages() -> None:
    from quarq.config import QuarqConfig
    from quarq.rag.retriever import Retriever

    cfg = QuarqConfig()
    cfg.rag.rerank_top_n = 3
    store = MagicMock()
    store.query.return_value = [
        _chunk("p1 first", 0.95, page=1),
        _chunk("p2", 0.90, page=2),
        _chunk("p1 again", 0.85, page=1),
        _chunk("p9 tail", 0.80, page=9),  # beyond top_n: keeps its embedding order
    ]
    embedder = MagicMock()
    embedder.embed_query.return_value = [0.1]
    reranker = MagicMock()
    # the re-ranker puts page 2 first, then the second chunk of page 1
    reranker.rerank.side_effect = lambda q, chunks: [chunks[1], chunks[2], chunks[0]]

    retriever = Retriever(store, embedder, cfg, reranker=reranker)
    results = retriever.retrieve("question", k=3)

    head = reranker.rerank.call_args.args[1]
    assert [c.content for c in head] == ["p1 first", "p2", "p1 again"]
    assert [(c.page, c.content) for c in results] == [(2, "p2"), (1, "p1 again"), (9, "p9 tail")]


def test_build_retriever_follows_config(fake_ce) -> None:
    from quarq.config import QuarqConfig
    from quarq.rag.reranker import Reranker
    from quarq.rag.retriever import build_retriever

    on = QuarqConfig()
    off = QuarqConfig()
    off.rag.rerank = False

    assert isinstance(build_retriever(MagicMock(), MagicMock(), on)._reranker, Reranker)
    assert build_retriever(MagicMock(), MagicMock(), off)._reranker is None
    assert fake_ce.loads == 0
