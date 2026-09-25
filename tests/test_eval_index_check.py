"""Tests for exact search (ExactStore) and the eval's index check."""

from __future__ import annotations

from pathlib import Path

import pytest

from quarq.rag.loader import Document

# Unit vectors in 3-d; similarity to the query [1, 0, 0] is the first coordinate.
VECTORS = {
    "a1": ("a.pdf", 1, "ecb_fsr", [1.0, 0.0, 0.0]),     # 1.00
    "a2": ("a.pdf", 2, "ecb_fsr", [0.9, 0.436, 0.0]),   # 0.90
    "b1": ("b.pdf", 1, "bdf_fsr", [0.8, 0.6, 0.0]),     # 0.80
    "b2": ("b.pdf", 2, "bdf_fsr", [0.8, 0.0, 0.6]),     # 0.80 (tie with b1)
    "c1": ("c.pdf", 1, "bdf_fsr", [0.1, 0.995, 0.0]),   # 0.10
}
QUERY = [1.0, 0.0, 0.0]


def _real_store(tmp_path: Path):
    from quarq.config import QuarqConfig
    from quarq.rag.store import VectorStore

    cfg = QuarqConfig()
    cfg.rag.chroma_path = str(tmp_path / "chroma")
    store = VectorStore(cfg)
    docs, embs = [], []
    for cid, (src, page, dtype, vec) in VECTORS.items():
        docs.append(Document(content=f"text {cid}", metadata={
            "source": src, "page": page, "doc_type": dtype, "date": "2025", "chunk_id": cid}))
        embs.append(vec)
    store.upsert(docs, embs)
    return store


# ---------------------------------------------------------------------------
# ExactStore
# ---------------------------------------------------------------------------


def test_exact_store_ranks_by_cosine(tmp_path: Path) -> None:
    from quarq.rag.exact import ExactStore

    exact = ExactStore(_real_store(tmp_path))
    chunks = exact.query(QUERY, k=3)

    assert [c.metadata["chunk_id"] for c in chunks][:2] == ["a1", "a2"]
    assert chunks[0].similarity == pytest.approx(1.0)
    assert chunks[1].similarity == pytest.approx(0.9, abs=1e-3)
    assert chunks[0].content == "text a1"
    assert (chunks[0].source, chunks[0].page) == ("a.pdf", 1)
    assert exact.count() == len(VECTORS)


def test_exact_store_applies_equality_filters(tmp_path: Path) -> None:
    from quarq.rag.exact import ExactStore

    exact = ExactStore(_real_store(tmp_path))
    chunks = exact.query(QUERY, k=5, filters={"doc_type": "bdf_fsr"})

    assert {c.metadata["chunk_id"] for c in chunks} == {"b1", "b2", "c1"}


def test_exact_store_rejects_non_equality_filters(tmp_path: Path) -> None:
    from quarq.exceptions import RAGError
    from quarq.rag.exact import ExactStore

    exact = ExactStore(_real_store(tmp_path))
    with pytest.raises(RAGError, match="equality"):
        exact.query(QUERY, k=2, filters={"page": {"$gte": 2}})


# ---------------------------------------------------------------------------
# missed_neighbours: tie-aware comparison of an index result with exact search
# ---------------------------------------------------------------------------


def _chunk(cid: str, sim: float):
    from quarq.rag.store import RetrievedChunk

    src, page, dtype, _ = VECTORS[cid]
    return RetrievedChunk(content="", metadata={"chunk_id": cid, "source": src, "page": page,
                                                "doc_type": dtype},
                          similarity=sim, source=src, page=page)


def test_missed_neighbours_counts_higher_scoring_chunks_left_out() -> None:
    from quarq.eval.index_check import missed_neighbours

    exact = [_chunk("a1", 1.0), _chunk("a2", 0.9), _chunk("b1", 0.8)]
    index = [_chunk("a1", 1.0), _chunk("b1", 0.8), _chunk("c1", 0.1)]  # a2 missing

    assert missed_neighbours(index, exact) == 1


def test_missed_neighbours_ignores_ties_at_the_boundary() -> None:
    from quarq.eval.index_check import missed_neighbours

    exact = [_chunk("a1", 1.0), _chunk("a2", 0.9), _chunk("b1", 0.8)]
    index = [_chunk("a1", 1.0), _chunk("a2", 0.9), _chunk("b2", 0.8)]  # b2 ties b1

    assert missed_neighbours(index, exact) == 0


# ---------------------------------------------------------------------------
# check_index end to end, with a deliberately lossy index
# ---------------------------------------------------------------------------


class _LossyStore:
    """Wraps a real store but drops one chunk from every result, like a weak index."""

    def __init__(self, store, drop: str) -> None:
        self._store, self._drop = store, drop

    def query(self, embedding, k=5, filters=None):
        return [c for c in self._store.query(embedding, k=k, filters=filters)
                if c.metadata["chunk_id"] != self._drop]

    def count(self) -> int:
        return self._store.count()


class _FixedEmbedder:
    def embed_query(self, query: str) -> list[float]:
        return QUERY


def _gold():
    from quarq.eval.dataset import GoldItem

    return [GoldItem(id="q1", question="anything", gold=[{"source": "a.pdf", "page": 2}],
                     doc_type="ecb_fsr", provenance="hand-written")]


def test_check_index_reports_a_perfect_index(tmp_path: Path) -> None:
    from quarq.config import QuarqConfig
    from quarq.eval.index_check import check_index
    from quarq.rag.exact import ExactStore

    store = _real_store(tmp_path)
    cfg = QuarqConfig()
    result = check_index(_gold(), store, ExactStore(store), _FixedEmbedder(), cfg,
                         max_k=2, use_doc_type_filter=False)

    assert result.n_questions == 1
    assert result.questions_with_gap == 0
    assert result.missed_chunks == 0
    assert result.exact_match_questions == 1


def test_check_index_detects_a_lossy_index(tmp_path: Path) -> None:
    from quarq.config import QuarqConfig
    from quarq.eval.index_check import check_index
    from quarq.rag.exact import ExactStore

    store = _real_store(tmp_path)
    cfg = QuarqConfig()
    result = check_index(_gold(), _LossyStore(store, drop="a2"), ExactStore(store),
                         _FixedEmbedder(), cfg, max_k=2, use_doc_type_filter=False)

    assert result.questions_with_gap == 1
    assert result.missed_chunks == 1
    assert result.exact_match_questions == 0  # a.pdf p2 drops out of the final top 2


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def test_markdown_and_json_include_the_index_check(tmp_path: Path) -> None:
    import json

    from quarq.eval.index_check import IndexCheck
    from quarq.eval.reporter import to_json, to_markdown
    from tests.test_eval_reporter import _result

    result = _result()
    result.index_check = IndexCheck(n_questions=2, candidates=20, questions_with_gap=1,
                                    missed_chunks=3, exact_match_questions=2)
    md = to_markdown(result, tmp_path / "r.md").read_text()
    data = json.loads(to_json(result, tmp_path / "r.json").read_text())

    assert "## Index check" in md
    assert "1 / 2" in md and "3" in md
    assert data["index_check"]["missed_chunks"] == 3
