"""Tests for quarq RAG and LLM layers."""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_retrieved_chunk(content: str, similarity: float, source: str = "test.pdf", page: int = 1):
    """Build a RetrievedChunk without importing the module (used before implementation)."""
    from quarq.rag.store import RetrievedChunk

    return RetrievedChunk(
        content=content,
        metadata={"source": source, "page": page, "doc_type": "macro", "date": "2025-01-01",
                   "chunk_id": hashlib.sha256(content.encode()).hexdigest()},
        similarity=similarity,
        source=source,
        page=page,
    )


def _make_config():
    from quarq.config import QuarqConfig
    return QuarqConfig()


# ---------------------------------------------------------------------------
# Module 1 — loader
# ---------------------------------------------------------------------------


def test_loader_chunks_pdf(tmp_path: Path) -> None:
    """load_pdf returns Document list with all 5 required metadata fields."""
    from quarq.rag.loader import load_pdf

    # Create a minimal PDF using pdfplumber-compatible approach via reportlab/fpdf
    # We mock pdfplumber.open to return synthetic pages
    mock_page = MagicMock()
    mock_page.extract_text.return_value = "This is a test document about the ECB Financial Stability Review. " * 20
    mock_page.page_number = 1

    mock_pdf = MagicMock()
    mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
    mock_pdf.__exit__ = MagicMock(return_value=False)
    mock_pdf.pages = [mock_page]
    mock_pdf.metadata = {}

    pdf_path = tmp_path / "ecb_stability_2025.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")

    with patch("pdfplumber.open", return_value=mock_pdf):
        docs = load_pdf(pdf_path)

    assert len(docs) >= 1
    for doc in docs:
        assert "source" in doc.metadata
        assert "doc_type" in doc.metadata
        assert "date" in doc.metadata
        assert "page" in doc.metadata
        assert "chunk_id" in doc.metadata
        # chunk_id must be a valid sha256 hex string (64 chars)
        assert len(doc.metadata["chunk_id"]) == 64


def test_loader_infers_doc_type_from_filename(tmp_path: Path) -> None:
    """load_pdf infers doc_type from filename patterns."""
    from quarq.rag.loader import load_pdf

    mock_page = MagicMock()
    mock_page.extract_text.return_value = "Sample text content for testing. " * 10
    mock_page.page_number = 1

    mock_pdf = MagicMock()
    mock_pdf.__enter__ = MagicMock(return_value=mock_pdf)
    mock_pdf.__exit__ = MagicMock(return_value=False)
    mock_pdf.pages = [mock_page]
    mock_pdf.metadata = {}

    cases = [
        ("ecb_stability_2025.pdf", "ecb_fsr"),
        ("amf_sfdr_template.pdf", "amf_sfdr"),
        ("unknown_file.pdf", "macro"),
    ]

    for filename, expected_doc_type in cases:
        pdf_path = tmp_path / filename
        pdf_path.write_bytes(b"%PDF-1.4 fake")
        with patch("pdfplumber.open", return_value=mock_pdf):
            docs = load_pdf(pdf_path)
        assert len(docs) >= 1
        assert docs[0].metadata["doc_type"] == expected_doc_type, (
            f"{filename} should map to {expected_doc_type}, got {docs[0].metadata['doc_type']}"
        )


# ---------------------------------------------------------------------------
# Module 2 — embedder
# ---------------------------------------------------------------------------


def test_embedder_adds_passage_prefix() -> None:
    """embed() adds 'passage: ' prefix; embed_query() adds 'query: ' prefix."""
    from quarq.rag.embedder import Embedder

    captured_calls: list[list[str]] = []

    def mock_encode(texts, batch_size=32, **kwargs):
        captured_calls.append(list(texts))
        return [[0.1, 0.2, 0.3]] * len(texts)

    embedder = Embedder(model_name="intfloat/multilingual-e5-large")

    mock_model = MagicMock()
    mock_model.encode.side_effect = mock_encode
    embedder._model = mock_model  # inject to bypass lazy load

    embedder.embed(["hello world", "second text"])
    assert captured_calls[0][0] == "passage: hello world"
    assert captured_calls[0][1] == "passage: second text"

    captured_calls.clear()
    embedder.embed_query("what is inflation?")
    assert captured_calls[0][0] == "query: what is inflation?"


# ---------------------------------------------------------------------------
# Module 3 — retriever
# ---------------------------------------------------------------------------


def test_retriever_filters_below_min_similarity() -> None:
    """retrieve() drops chunks below min_similarity threshold."""
    from quarq.rag.retriever import Retriever

    cfg = _make_config()
    mock_store = MagicMock()
    mock_embedder = MagicMock()
    mock_embedder.embed_query.return_value = [0.1, 0.2, 0.3]

    chunks = [
        _make_retrieved_chunk("high relevance", 0.8),
        _make_retrieved_chunk("low relevance", 0.3),
        _make_retrieved_chunk("very low", 0.2),
    ]
    mock_store.query.return_value = chunks

    retriever = Retriever(store=mock_store, embedder=mock_embedder, cfg=cfg)
    results = retriever.retrieve("test query", min_similarity=0.35)

    assert len(results) == 1
    assert results[0].content == "high relevance"
    assert results[0].similarity == 0.8


def test_retriever_returns_empty_when_no_results() -> None:
    """retrieve() returns [] without raising when store returns no chunks."""
    from quarq.rag.retriever import Retriever

    cfg = _make_config()
    mock_store = MagicMock()
    mock_embedder = MagicMock()
    mock_embedder.embed_query.return_value = [0.1, 0.2, 0.3]
    mock_store.query.return_value = []

    retriever = Retriever(store=mock_store, embedder=mock_embedder, cfg=cfg)
    results = retriever.retrieve("test query")

    assert results == []


# ---------------------------------------------------------------------------
# Module 4 — generator
# ---------------------------------------------------------------------------


def test_generator_builds_correct_prompt() -> None:
    """answer() prompt contains RETRIEVED DOCUMENTS section and query text."""
    from quarq.rag.generator import answer

    cfg = _make_config()
    chunks = [_make_retrieved_chunk("ECB policy rate is 4.0 percent.", 0.9)]

    captured_prompts: list[str] = []

    mock_llm = MagicMock()
    mock_llm.generate.side_effect = lambda prompt, system="": (
        captured_prompts.append(prompt) or "Test answer."
    )

    with patch("quarq.rag.generator.get_llm", return_value=mock_llm):
        answer("What is the ECB rate?", chunks, cfg)

    assert len(captured_prompts) == 1
    prompt = captured_prompts[0]
    assert "RETRIEVED DOCUMENTS" in prompt
    assert "ECB policy rate is 4.0 percent." in prompt
    assert "What is the ECB rate?" in prompt


def test_generator_returns_generation_result() -> None:
    """answer() returns a GenerationResult with the LLM's answer text."""
    from quarq.llm.base import GenerationResult
    from quarq.rag.generator import answer

    cfg = _make_config()
    chunks = [_make_retrieved_chunk("Some context.", 0.85)]

    mock_llm = MagicMock()
    mock_llm.generate.return_value = "test answer"
    mock_llm.model = "test-model"
    mock_llm.name = "test-backend"

    with patch("quarq.rag.generator.get_llm", return_value=mock_llm):
        result = answer("What happened?", chunks, cfg)

    assert isinstance(result, GenerationResult)
    assert result.answer == "test answer"


# ---------------------------------------------------------------------------
# Module 5 — LLM backend fallback
# ---------------------------------------------------------------------------


def test_lmstudio_unavailable_falls_back_to_claude() -> None:
    """get_llm() returns ClaudeLLM when LM Studio is not available."""
    from quarq.llm.lmstudio import get_llm

    cfg = _make_config()

    with (
        patch("quarq.llm.lmstudio.is_available", return_value=False),
        patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test-key"}),
        patch("quarq.llm.claude.anthropic") as mock_anthropic,
    ):
        mock_anthropic.Anthropic.return_value = MagicMock()
        llm = get_llm(cfg, agent="research")

    from quarq.llm.claude import ClaudeLLM
    assert isinstance(llm, ClaudeLLM)


def test_neither_backend_raises_rag_error() -> None:
    """get_llm() raises RAGError when neither LM Studio nor Claude API is available."""
    from quarq.exceptions import RAGError
    from quarq.llm.lmstudio import get_llm

    cfg = _make_config()

    env_without_key = {k: v for k, v in __import__("os").environ.items()
                       if k != "ANTHROPIC_API_KEY"}

    with (
        patch("quarq.llm.lmstudio.is_available", return_value=False),
        patch.dict("os.environ", env_without_key, clear=True),
    ):
        with pytest.raises(RAGError, match="No LLM backend available"):
            get_llm(cfg, agent="research")
