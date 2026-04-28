"""FastAPI endpoint tests using TestClient. No real network calls."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from quarq.api.app import app
from quarq.llm.base import GenerationResult
from quarq.rag.store import RetrievedChunk
from quarq.status import StatusResult


def _make_price_df(ticker: str, rows: int = 30) -> pd.DataFrame:
    """Build a minimal price DataFrame matching EquityProvider output."""
    idx = pd.date_range("2024-01-02", periods=rows, freq="B")
    return pd.DataFrame(
        {"value": [100.0 + i for i in range(rows)], "series_id": ticker, "source": "equity"},
        index=idx,
    )


def _make_mock_status(online: bool = True) -> StatusResult:
    """Build a StatusResult with all services online (or all offline)."""
    return StatusResult(
        lmstudio_online=online,
        lmstudio_models=["qwen/qwen3.5-9b"],
        lmstudio_active_model="qwen/qwen3.5-9b",
        fred_status="connected",
        ecb_online=online,
        corpus_docs=5,
        corpus_chunks=100,
    )


@pytest.fixture
def client():
    """TestClient with mocked lifespan (no real startup checks)."""
    mock_status = _make_mock_status(online=True)
    from quarq.config import QuarqConfig

    mock_cfg = QuarqConfig()

    with patch("quarq.api.app.run_status_checks", return_value=mock_status), \
         patch("quarq.api.app.load_config", return_value=mock_cfg):
        with TestClient(app) as c:
            yield c


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------

def test_root_returns_name_and_version(client):
    """GET / returns 200 with name='quarq' and a version field."""
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "quarq"
    assert "version" in data


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

def test_health_returns_ok_structure(client):
    """GET /health returns 200 with a status field."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data
    assert data["status"] in ("ok", "degraded", "offline")


# ---------------------------------------------------------------------------
# Portfolio
# ---------------------------------------------------------------------------

def test_portfolio_metrics_validates_weights(client):
    """POST /portfolio/metrics with weights not summing to 1.0 returns 422."""
    payload = {
        "tickers": ["MC.PA"],
        "weights": [0.5],  # sum = 0.5, not 1.0
        "start": "2024-01-01",
        "end": "2024-12-31",
    }
    resp = client.post("/portfolio/metrics", json=payload)
    assert resp.status_code == 422


def test_portfolio_metrics_returns_metrics(client):
    """POST /portfolio/metrics with valid inputs returns metric fields."""
    prices = {
        "MC.PA": _make_price_df("MC.PA"),
        "^FCHI": _make_price_df("^FCHI"),
    }
    with patch("quarq.api.routes.portfolio.fetch_portfolio", return_value=prices), \
         patch("quarq.api.routes.portfolio.get_risk_free_rate", return_value=0.03):
        payload = {
            "tickers": ["MC.PA"],
            "weights": [1.0],
            "start": "2024-01-01",
            "end": "2024-12-31",
            "benchmark": "^FCHI",
        }
        resp = client.post("/portfolio/metrics", json=payload)

    assert resp.status_code == 200
    data = resp.json()
    assert "sharpe_ratio" in data
    assert "max_drawdown" in data
    assert "cagr" in data


# ---------------------------------------------------------------------------
# RAG
# ---------------------------------------------------------------------------

def test_rag_query_empty_corpus_returns_422(client):
    """POST /rag/query returns 422 when corpus is empty."""
    with patch("quarq.api.routes.rag.VectorStore") as MockStore:
        MockStore.return_value.count.return_value = 0
        resp = client.post("/rag/query", json={"question": "What are the risks?"})
    assert resp.status_code == 422
    assert "empty" in resp.json()["detail"].lower()


def test_rag_query_returns_answer_and_sources(client):
    """POST /rag/query returns answer and sources when corpus has chunks."""
    mock_chunk = RetrievedChunk(
        content="French financial risks include...",
        metadata={
            "source": "ecb_fsr_2024.pdf",
            "page": 3,
            "doc_type": "ecb_fsr",
            "date": "2024-01-01",
            "chunk_id": "abc123",
        },
        similarity=0.82,
        source="ecb_fsr_2024.pdf",
        page=3,
    )
    mock_result = GenerationResult(
        answer="The main risks are...",
        model="qwen/qwen3.6-27b",
        backend="lmstudio",
        latency_ms=450,
    )

    with patch("quarq.api.routes.rag.VectorStore") as MockStore, \
         patch("quarq.api.routes.rag.Embedder"), \
         patch("quarq.api.routes.rag.Retriever") as MockRetriever, \
         patch("quarq.api.routes.rag.answer", return_value=mock_result):
        MockStore.return_value.count.return_value = 100
        MockRetriever.return_value.retrieve.return_value = [mock_chunk, mock_chunk]
        resp = client.post("/rag/query", json={"question": "What are the risks?"})

    assert resp.status_code == 200
    data = resp.json()
    assert "answer" in data
    assert "sources" in data
    assert len(data["sources"]) == 2


def test_rag_add_nonexistent_path_returns_404(client):
    """POST /rag/add with a non-existent path returns 404."""
    resp = client.post("/rag/add", json={"path": "/nonexistent/path/to/docs"})
    assert resp.status_code == 404


def test_rag_corpus_delete_requires_confirm_header(client):
    """DELETE /rag/corpus without X-Confirm-Delete header returns 400."""
    resp = client.delete("/rag/corpus")
    assert resp.status_code == 400
    assert "X-Confirm-Delete" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def test_data_unknown_provider_returns_404(client):
    """GET /data/unknown_provider/... returns 404."""
    resp = client.get(
        "/data/unknown_provider/SERIES_ID",
        params={"start": "2024-01-01", "end": "2024-12-31"},
    )
    assert resp.status_code == 404


def test_data_fetch_returns_correct_schema(client):
    """GET /data/equity/MC.PA returns 200 with series_id, provider, data fields."""
    df = _make_price_df("MC.PA")
    with patch("quarq.api.routes.data.get_provider") as mock_get:
        mock_prov = MagicMock()
        mock_prov.fetch.return_value = df
        mock_get.return_value = mock_prov
        resp = client.get(
            "/data/equity/MC.PA",
            params={"start": "2024-01-01", "end": "2024-12-31"},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["series_id"] == "MC.PA"
    assert data["provider"] == "equity"
    assert isinstance(data["data"], list)
    assert len(data["data"]) > 0
