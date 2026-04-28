"""Pydantic v2 request and response models for the quarq API."""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, model_validator


class PortfolioRequest(BaseModel):
    """Request body for POST /portfolio/metrics."""

    tickers: list[str]
    weights: list[float]
    start: date
    end: date
    benchmark: str = "^FCHI"
    currency: str = "EUR"

    @model_validator(mode="after")
    def weights_sum_to_one(self) -> "PortfolioRequest":
        """Validate that weights sum to 1.0 within 0.01 tolerance.

        Args:
            None (model_validator receives self).

        Returns:
            self if valid.

        Raises:
            ValueError: If abs(sum(weights) - 1.0) > 0.01.
        """
        total = sum(self.weights)
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"weights must sum to 1.0 (got {total:.4f})")
        return self


class RAGQueryRequest(BaseModel):
    """Request body for POST /rag/query."""

    question: str
    doc_type: str | None = None
    k: int = 5
    portfolio_context: str = ""


class RAGAddRequest(BaseModel):
    """Request body for POST /rag/add."""

    path: str
    recursive: bool = True


class HealthResponse(BaseModel):
    """Response for GET /health."""

    status: str  # "ok" | "degraded" | "offline"
    version: str
    lmstudio_online: bool
    lmstudio_model: str
    fred_status: str
    ecb_online: bool
    corpus_chunks: int
    corpus_docs: int


class SourceCitation(BaseModel):
    """A single source citation returned in a RAG answer."""

    source: str
    page: int
    similarity: float


class MetricsResponse(BaseModel):
    """Response for POST /portfolio/metrics."""

    tickers: list[str]
    weights: list[float]
    start: date
    end: date
    benchmark: str
    sharpe_ratio: float | None
    max_drawdown: float | None
    cagr: float | None
    volatility: float | None
    var_95: float | None
    beta: float | None
    alpha: float | None
    risk_free_rate: float
    computation_time_ms: int


class RAGQueryResponse(BaseModel):
    """Response for POST /rag/query."""

    question: str
    answer: str
    sources: list[SourceCitation]
    model: str
    backend: str
    latency_ms: int


class RAGAddResponse(BaseModel):
    """Response for POST /rag/add."""

    path: str
    chunks_added: int
    chunks_skipped: int
    total_chunks: int
    total_docs: int


class DataFetchResponse(BaseModel):
    """Response for GET /data/{provider}/{series_id}."""

    series_id: str
    provider: str
    start: date
    end: date
    records: int
    data: list[dict]  # [{"date": "YYYY-MM-DD", "value": float}]
    cached: bool
