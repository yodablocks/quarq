"""Tests for VectorStore.list_chunks (ChromaDB collection mocked)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from quarq.exceptions import RAGError
from quarq.rag.store import VectorStore


def _store(collection: MagicMock) -> VectorStore:
    store = VectorStore.__new__(VectorStore)  # skip __init__: no ChromaDB on disk
    store._collection = collection
    return store


def test_list_chunks_maps_documents_and_metadata() -> None:
    collection = MagicMock()
    collection.get.return_value = {
        "documents": ["text one", "text two"],
        "metadatas": [
            {"source": "a.pdf", "page": 3, "doc_type": "ecb_fsr", "chunk_id": "c1"},
            {"source": "b.pdf", "page": "7", "doc_type": "amf_sfdr", "chunk_id": "c2"},
        ],
    }

    chunks = _store(collection).list_chunks()

    collection.get.assert_called_once_with(include=["documents", "metadatas"])
    assert [(c.source, c.page, c.content) for c in chunks] == [
        ("a.pdf", 3, "text one"),
        ("b.pdf", 7, "text two"),
    ]
    assert chunks[0].metadata["doc_type"] == "ecb_fsr"
    assert chunks[0].similarity == 0.0


def test_list_chunks_wraps_chroma_errors() -> None:
    collection = MagicMock()
    collection.get.side_effect = RuntimeError("boom")

    with pytest.raises(RAGError, match="list_chunks failed"):
        _store(collection).list_chunks()
