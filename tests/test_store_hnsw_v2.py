"""Tests for the v2 collection: explicit HNSW settings, config drift, and migration from v1."""

from __future__ import annotations

import logging
from pathlib import Path

import chromadb
import pytest

LEGACY = "quarq_rag_v1"


def _cfg(tmp_path: Path):
    from quarq.config import QuarqConfig

    cfg = QuarqConfig()
    cfg.rag.chroma_path = str(tmp_path / "chroma")
    return cfg


def _hnsw(collection) -> dict:
    return collection.configuration_json["hnsw"]


def _make_legacy(tmp_path: Path, n: int = 3) -> None:
    """Create a v1 collection the way quarq used to: default HNSW, cosine via metadata."""
    client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
    col = client.get_or_create_collection(name=LEGACY, metadata={"hnsw:space": "cosine"})
    col.add(
        ids=[f"id{i}" for i in range(n)],
        embeddings=[[float(i + 1), 0.5, 0.25] for i in range(n)],
        documents=[f"text {i}" for i in range(n)],
        metadatas=[{"source": "a.pdf", "page": i + 1, "doc_type": "ecb_fsr", "date": "2025",
                    "chunk_id": f"id{i}", "period_start": "2025-01-01"} for i in range(n)],
    )


def test_new_store_uses_the_configured_hnsw_settings(tmp_path: Path) -> None:
    from quarq.constants import RAG_COLLECTION_NAME, RAG_HNSW_CONFIG
    from quarq.rag.store import VectorStore

    store = VectorStore(_cfg(tmp_path))

    assert store._collection.name == RAG_COLLECTION_NAME
    hnsw = _hnsw(store._collection)
    for key, value in RAG_HNSW_CONFIG.items():
        assert hnsw[key] == value, key
    assert store.config_drift == {}


def test_config_drift_is_reported(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    from quarq.constants import RAG_COLLECTION_NAME
    from quarq.rag.store import VectorStore

    client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
    client.create_collection(
        name=RAG_COLLECTION_NAME, configuration={"hnsw": {"space": "cosine", "ef_search": 50}}
    )

    with caplog.at_level(logging.WARNING, logger="quarq.rag.store"):
        store = VectorStore(_cfg(tmp_path))

    assert "ef_search" in store.config_drift
    assert "ef_search" in caplog.text


def test_legacy_collection_counts(tmp_path: Path) -> None:
    from quarq.rag.store import VectorStore

    _make_legacy(tmp_path, n=3)
    store = VectorStore(_cfg(tmp_path))

    assert store.legacy_collection_counts() == {LEGACY: 3}


def test_migrate_copies_everything_without_reembedding(tmp_path: Path) -> None:
    from quarq.rag.store import VectorStore

    _make_legacy(tmp_path, n=3)
    store = VectorStore(_cfg(tmp_path))

    copied = store.migrate_from(LEGACY, batch_size=2)  # batches smaller than the collection

    assert copied == 3
    legacy = chromadb.PersistentClient(path=str(tmp_path / "chroma")).get_collection(LEGACY)
    old = legacy.get(include=["embeddings", "documents", "metadatas"])
    new = store._collection.get(ids=old["ids"], include=["embeddings", "documents", "metadatas"])
    def by_id(got: dict) -> dict:
        rows = zip(got["ids"], got["embeddings"], got["documents"], got["metadatas"], strict=True)
        return {i: (list(e), d, m) for i, e, d, m in rows}

    by_id_old, by_id_new = by_id(old), by_id(new)
    assert by_id_new == by_id_old
    assert legacy.count() == 3  # the legacy collection is left untouched


def test_migrate_is_idempotent(tmp_path: Path) -> None:
    from quarq.rag.store import VectorStore

    _make_legacy(tmp_path, n=3)
    store = VectorStore(_cfg(tmp_path))

    store.migrate_from(LEGACY)
    store.migrate_from(LEGACY)

    assert store.count() == 3


def test_migrate_from_missing_collection_raises(tmp_path: Path) -> None:
    from quarq.exceptions import RAGError
    from quarq.rag.store import VectorStore

    store = VectorStore(_cfg(tmp_path))
    with pytest.raises(RAGError, match="quarq_rag_v0"):
        store.migrate_from("quarq_rag_v0")


def test_reset_keeps_the_hnsw_settings(tmp_path: Path) -> None:
    from quarq.constants import RAG_HNSW_CONFIG
    from quarq.rag.store import VectorStore

    store = VectorStore(_cfg(tmp_path))
    store.reset()

    assert _hnsw(store._collection)["ef_search"] == RAG_HNSW_CONFIG["ef_search"]
