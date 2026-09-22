"""Contract tests for the Open WebUI tool classes in demo/tools/.

These mock the HTTP layer so the tools are exercised without quarq serve,
Docker, or LM Studio running. Their purpose is to catch drift between the
tool request/response shapes and the FastAPI models in quarq/api/models.py.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from tools.quarq_portfolio_tool import Tools as PortfolioTools
from tools.quarq_rag_tool import Tools as RagTools

from quarq.api.models import MetricsResponse, PortfolioRequest, RAGQueryRequest


class _FakeResponse:
    """Minimal stand-in for httpx.Response."""

    def __init__(self, payload: dict[str, Any], status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def json(self) -> dict[str, Any]:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            request = httpx.Request("POST", "http://testserver/rag/query")
            response = httpx.Response(self.status_code, request=request)
            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code}", request=request, response=response
            )


@pytest.fixture
def captured(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Capture the outgoing request and return a configurable payload."""
    box: dict[str, Any] = {"payload": {}, "status": 200}

    def fake_post(url: str, json: dict[str, Any], timeout: float) -> _FakeResponse:
        box["url"] = url
        box["json"] = json
        return _FakeResponse(box["payload"], box["status"])

    monkeypatch.setattr(httpx, "post", fake_post)
    return box


# --- base URL configuration -------------------------------------------------


def test_base_url_defaults_to_docker_host(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without QUARQ_API_URL the tools target the Docker-internal host."""
    monkeypatch.delenv("QUARQ_API_URL", raising=False)
    assert RagTools().base_url == "http://host.docker.internal:8000"
    assert PortfolioTools().base_url == "http://host.docker.internal:8000"


def test_base_url_honours_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """QUARQ_API_URL overrides the default so the tools work from the host."""
    monkeypatch.setenv("QUARQ_API_URL", "http://127.0.0.1:8000")
    assert RagTools().base_url == "http://127.0.0.1:8000"
    assert PortfolioTools().base_url == "http://127.0.0.1:8000"


# --- RAG tool contract ------------------------------------------------------


def test_rag_request_body_matches_api_model(captured: dict[str, Any]) -> None:
    """The RAG tool's request body validates against RAGQueryRequest."""
    captured["payload"] = {"answer": "Yes.", "sources": []}
    RagTools().query_financial_documents("What is concentration risk?")

    assert captured["url"].endswith("/rag/query")
    body = captured["json"]
    # Regression: the tool previously sent "n_results", which Pydantic silently
    # dropped, so the requested page size never reached the retriever.
    assert "n_results" not in body
    assert body["k"] == 5
    parsed = RAGQueryRequest.model_validate(body)
    assert parsed.question == "What is concentration risk?"
    assert parsed.k == 5


def test_rag_formats_answer_with_sources(captured: dict[str, Any]) -> None:
    """Citations are appended to the answer text."""
    captured["payload"] = {
        "answer": "Concentration is elevated.",
        "sources": [
            {"source": "ecb_fsr_2025.pdf", "page": 42, "similarity": 0.81},
            {"source": "amf_sfdr.pdf", "page": 7, "similarity": 0.55},
        ],
    }
    out = RagTools().query_financial_documents("q")

    assert "Concentration is elevated." in out
    assert "ecb_fsr_2025.pdf (page 42)" in out
    assert "amf_sfdr.pdf (page 7)" in out


def test_rag_answer_without_sources_is_returned_bare(captured: dict[str, Any]) -> None:
    """An empty source list yields the answer with no Sources block."""
    captured["payload"] = {"answer": "No corpus match.", "sources": []}
    out = RagTools().query_financial_documents("q")

    assert out == "No corpus match."
    assert "Sources:" not in out


def test_rag_reports_server_down(monkeypatch: pytest.MonkeyPatch) -> None:
    """A connection error produces an actionable message, not a traceback."""

    def boom(*args: Any, **kwargs: Any) -> None:
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "post", boom)
    out = RagTools().query_financial_documents("q")

    assert "not running" in out
    assert "quarq serve" in out


# --- Portfolio tool contract ------------------------------------------------


def _metrics_payload(**overrides: Any) -> dict[str, Any]:
    """Build a MetricsResponse-shaped payload, serialised as the API would."""
    base = MetricsResponse(
        tickers=["MC.PA", "TTE.PA"],
        weights=[0.5, 0.5],
        start="2025-01-01",  # type: ignore[arg-type]
        end="2025-12-31",  # type: ignore[arg-type]
        benchmark="^FCHI",
        sharpe_ratio=1.23,
        max_drawdown=-0.184,
        cagr=0.0972,
        volatility=0.151,
        var_95=-0.021,
        beta=0.94,
        alpha=0.013,
    )
    payload = base.model_dump(mode="json")
    payload.update(overrides)
    return payload


def test_portfolio_request_body_matches_api_model(captured: dict[str, Any]) -> None:
    """The portfolio tool's request body validates against PortfolioRequest."""
    captured["payload"] = _metrics_payload()
    PortfolioTools().analyse_portfolio(
        tickers="MC.PA,TTE.PA", weights="0.5,0.5", benchmark="^FCHI"
    )

    assert captured["url"].endswith("/portfolio/metrics")
    parsed = PortfolioRequest.model_validate(captured["json"])
    assert parsed.tickers == ["MC.PA", "TTE.PA"]
    assert parsed.weights == [0.5, 0.5]
    assert parsed.benchmark == "^FCHI"


def test_portfolio_renders_sharpe_from_correct_key(captured: dict[str, Any]) -> None:
    """Sharpe is read from sharpe_ratio, the field the API actually returns.

    Regression: the tool read m["sharpe"], which never exists, so Sharpe was
    always displayed as 0.00 in Open WebUI.
    """
    captured["payload"] = _metrics_payload(sharpe_ratio=1.23)
    out = PortfolioTools().analyse_portfolio(tickers="MC.PA", weights="1.0")

    assert "Sharpe:        1.23" in out
    assert "Sharpe:        0.00" not in out


def test_portfolio_renders_all_metrics(captured: dict[str, Any]) -> None:
    """Every metric in the response is rendered with the expected scaling."""
    captured["payload"] = _metrics_payload()
    out = PortfolioTools().analyse_portfolio(tickers="MC.PA,TTE.PA", weights="0.5,0.5")

    assert "CAGR:          9.72%" in out
    assert "Max Drawdown:  -18.40%" in out
    assert "VaR 95 (daily):-2.10%" in out
    assert "Volatility:    15.10%" in out
    assert "Beta:          0.94" in out
    assert "Alpha:         1.30%" in out


def test_portfolio_handles_null_beta_and_alpha(captured: dict[str, Any]) -> None:
    """Null metrics render as n/a rather than crashing.

    beta and alpha are None whenever benchmark data is unavailable; the old
    .get(key, 0) default did not guard against an explicitly null value.
    """
    captured["payload"] = _metrics_payload(beta=None, alpha=None)
    out = PortfolioTools().analyse_portfolio(tickers="MC.PA", weights="1.0")

    assert "Beta:          n/a" in out
    assert "Alpha:         n/a" in out
    assert "failed" not in out.lower()


def test_portfolio_handles_all_null_metrics(captured: dict[str, Any]) -> None:
    """A response with every metric null still renders a readable summary."""
    captured["payload"] = _metrics_payload(
        sharpe_ratio=None,
        max_drawdown=None,
        cagr=None,
        volatility=None,
        var_95=None,
        beta=None,
        alpha=None,
    )
    out = PortfolioTools().analyse_portfolio(tickers="MC.PA", weights="1.0")

    assert out.count("n/a") == 7
    assert "failed" not in out.lower()


def test_portfolio_includes_narrative_when_present(captured: dict[str, Any]) -> None:
    """A narrative field is surfaced under a Commentary heading."""
    captured["payload"] = _metrics_payload(narrative="Solid risk-adjusted return.")
    out = PortfolioTools().analyse_portfolio(tickers="MC.PA", weights="1.0")

    assert "Commentary:" in out
    assert "Solid risk-adjusted return." in out


def test_portfolio_defaults_dates_to_trailing_year(captured: dict[str, Any]) -> None:
    """Omitted dates default to a trailing 12-month window."""
    from datetime import date

    captured["payload"] = _metrics_payload()
    PortfolioTools().analyse_portfolio(tickers="MC.PA", weights="1.0")

    body = captured["json"]
    assert body["end"] == date.today().isoformat()
    assert body["start"] < body["end"]


def test_portfolio_rejects_malformed_weights(captured: dict[str, Any]) -> None:
    """Non-numeric weights produce an input error, not a traceback."""
    captured["payload"] = _metrics_payload()
    out = PortfolioTools().analyse_portfolio(tickers="MC.PA", weights="abc")

    assert "Invalid input" in out


def test_portfolio_reports_server_down(monkeypatch: pytest.MonkeyPatch) -> None:
    """A connection error produces an actionable message."""

    def boom(*args: Any, **kwargs: Any) -> None:
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "post", boom)
    out = PortfolioTools().analyse_portfolio(tickers="MC.PA", weights="1.0")

    assert "not running" in out
    assert "quarq serve" in out


def test_rag_explains_empty_corpus_on_422(captured: dict[str, Any]) -> None:
    """A 422 (empty corpus) yields actionable guidance, not a raw error.

    An unindexed corpus is a realistic first-run state in Open WebUI.
    """
    captured["payload"] = {"detail": "RAG corpus is empty."}
    captured["status"] = 422
    out = RagTools().query_financial_documents("q")

    assert "corpus is empty" in out.lower()
    assert "quarq rag add" in out


def test_rag_reports_other_http_errors(captured: dict[str, Any]) -> None:
    """Non-422 HTTP errors still surface as a readable failure string."""
    captured["payload"] = {"detail": "boom"}
    captured["status"] = 500
    out = RagTools().query_financial_documents("q")

    assert "RAG query failed" in out


def test_portfolio_reports_http_errors(captured: dict[str, Any]) -> None:
    """A 503 from the metrics endpoint surfaces as a readable failure string."""
    captured["payload"] = {"detail": "Equity data fetch failed"}
    captured["status"] = 503
    out = PortfolioTools().analyse_portfolio(tickers="MC.PA", weights="1.0")

    assert "Portfolio analysis failed" in out
