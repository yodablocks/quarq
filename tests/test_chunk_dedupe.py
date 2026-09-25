"""Tests for the chunker's overlap tail, replace-on-reindex, and deduplicating an index."""

from __future__ import annotations

import hashlib
from pathlib import Path

from quarq.rag.loader import Document

# ---------------------------------------------------------------------------
# Chunker: no tail chunk made only of overlap words
# ---------------------------------------------------------------------------


def _words(n: int) -> str:
    return " ".join(f"w{i}" for i in range(n))


def test_page_of_exactly_chunk_size_gives_one_chunk() -> None:
    from quarq.rag.loader import _chunk_text

    chunks = _chunk_text(_words(512), page_num=1, chunk_size=512, chunk_overlap=64)

    assert len(chunks) == 1
    assert len(chunks[0][0].split()) == 512


def test_tail_with_new_words_is_kept_with_its_overlap() -> None:
    from quarq.rag.loader import _chunk_text

    chunks = _chunk_text(_words(522), page_num=1, chunk_size=512, chunk_overlap=64)

    assert len(chunks) == 2
    tail = chunks[1][0].split()
    assert len(tail) == 64 + 10
    assert tail[-1] == "w521"


def test_short_page_gives_one_chunk() -> None:
    from quarq.rag.loader import _chunk_text

    chunks = _chunk_text(_words(40), page_num=1, chunk_size=512, chunk_overlap=64)

    assert [len(c[0].split()) for c in chunks] == [40]


def test_chunk_id_formula() -> None:
    from quarq.rag.loader import chunk_id

    expected = hashlib.sha256(b"a.pdf:3:some text").hexdigest()
    assert chunk_id("a.pdf", 3, "some text") == expected


# ---------------------------------------------------------------------------
# Store helpers (real ChromaDB in tmp_path)
# ---------------------------------------------------------------------------


def _store(tmp_path: Path):
    from quarq.config import QuarqConfig
    from quarq.rag.store import VectorStore

    cfg = QuarqConfig()
    cfg.rag.chroma_path = str(tmp_path / "chroma")
    return VectorStore(cfg)


def _doc(source: str, page: int, text: str, cid: str | None = None) -> Document:
    from quarq.rag.loader import chunk_id

    return Document(content=text, metadata={
        "source": source, "page": page, "doc_type": "ecb_fsr", "date": "2025",
        "chunk_id": cid or chunk_id(source, page, text)})


def _vec(i: int) -> list[float]:
    return [1.0, 0.1 * i, 0.01 * i]


def test_replace_sources_removes_chunks_a_reindex_no_longer_produces(tmp_path: Path) -> None:
    store = _store(tmp_path)
    old = [_doc("a.pdf", 1, "old text"), _doc("a.pdf", 2, "kept text"), _doc("b.pdf", 1, "other")]
    store.upsert(old, [_vec(i) for i in range(3)])

    new = [_doc("a.pdf", 2, "kept text"), _doc("a.pdf", 3, "new text")]
    added, removed = store.replace_sources(new, [_vec(5), _vec(6)])

    texts = set(store._collection.get(include=["documents"])["documents"])
    assert texts == {"kept text", "new text", "other"}  # b.pdf untouched
    assert (added, removed) == (2, 1)


def _legacy_duplicate_index(tmp_path: Path):
    """An index where a.pdf p1 is stored twice: once under the old text-only id."""
    store = _store(tmp_path)
    text = "duplicated paragraph " * 10
    legacy_id = hashlib.sha256(text.encode()).hexdigest()
    tail = " ".join(text.split()[-4:])  # an overlap-only tail, contained in the chunk
    docs = [
        _doc("a.pdf", 1, text),
        _doc("a.pdf", 1, text, cid=legacy_id),
        _doc("a.pdf", 1, tail),
        _doc("a.pdf", 2, "unique page two"),
        _doc("b.pdf", 1, text),  # same text in another document is NOT a duplicate
    ]
    store.upsert(docs, [_vec(i) for i in range(len(docs))])
    return store, text, legacy_id, tail


def test_find_duplicates_keeps_the_current_id_and_flags_contained_tails(tmp_path: Path) -> None:
    from quarq.rag.loader import chunk_id

    store, text, legacy_id, tail = _legacy_duplicate_index(tmp_path)
    report = store.find_duplicate_chunks(max_tail_words=4)

    assert report.exact_duplicates == [legacy_id]
    assert report.contained_tails == [chunk_id("a.pdf", 1, tail)]


def test_remove_chunks_deletes_only_the_given_ids(tmp_path: Path) -> None:
    from quarq.rag.loader import chunk_id

    store, text, legacy_id, tail = _legacy_duplicate_index(tmp_path)
    report = store.find_duplicate_chunks(max_tail_words=4)

    removed = store.remove_chunks(report.exact_duplicates + report.contained_tails)

    remaining = set(store._collection.get()["ids"])
    assert removed == 2
    assert remaining == {chunk_id("a.pdf", 1, text), chunk_id("a.pdf", 2, "unique page two"),
                         chunk_id("b.pdf", 1, text)}
    again = store.find_duplicate_chunks(max_tail_words=4)
    assert again.exact_duplicates == [] and again.contained_tails == []
