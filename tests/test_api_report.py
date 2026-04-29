"""API tests for GET /report and POST /report endpoints."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from quarq.api.app import app
from quarq.status import StatusResult


def _make_price_df(ticker: str, rows: int = 60) -> pd.DataFrame:
    idx = pd.date_range("2024-01-02", periods=rows, freq="B")
    return pd.DataFrame(
        {"value": [100.0 + i * 0.5 for i in range(rows)], "series_id": ticker, "source": "equity"},
        index=idx,
    )


def _make_mock_status() -> StatusResult:
    return StatusResult(
        lmstudio_online=False,
        lmstudio_models=[],
        lmstudio_active_model="",
        fred_status="no key",
        ecb_online=False,
        corpus_docs=0,
        corpus_chunks=0,
    )


@pytest.fixture
def client():
    """TestClient with mocked lifespan."""
    mock_status = _make_mock_status()
    from quarq.config import QuarqConfig

    mock_cfg = QuarqConfig()

    with patch("quarq.api.app.run_status_checks", return_value=mock_status), \
         patch("quarq.api.app.load_config", return_value=mock_cfg):
        with TestClient(app) as c:
            yield c


def _mock_prices(tickers: list[str]) -> dict[str, pd.DataFrame]:
    return {t: _make_price_df(t) for t in tickers}


def test_post_report_html_200(client) -> None:
    """POST /report returns 200 with text/html content containing Plotly."""
    body = {
        "tickers": ["MC.PA", "AIR.PA"],
        "weights": [0.5, 0.5],
        "start": "2024-01-01",
        "end": "2024-06-30",
        "benchmark": "^FCHI",
    }
    tickers_with_bench = ["MC.PA", "AIR.PA", "^FCHI"]
    mock_prices = _mock_prices(tickers_with_bench)

    with patch("quarq.api.routes.report.fetch_portfolio", return_value=mock_prices), \
         patch("quarq.api.routes.report.get_risk_free_rate", return_value=0.03):
        resp = client.post("/report?format=html", json=body)

    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "Portfolio" in resp.text
    assert "plotly" in resp.text.lower()


def test_get_report_json_200(tmp_path, client) -> None:
    """GET /report with a TOML portfolio returns 200."""
    toml_content = """\
name = "Test"
benchmark = "^FCHI"
start = "2024-01-01"
end = "2024-06-30"
currency = "EUR"

[[sleeve]]
name = "Equity"
weight = 1.0

[[sleeve.holding]]
ticker = "MC.PA"
weight = 1.0
"""
    p = tmp_path / "portfolio.toml"
    p.write_text(toml_content)

    mock_prices = _mock_prices(["MC.PA", "^FCHI"])

    with patch("quarq.api.routes.report.fetch_portfolio", return_value=mock_prices), \
         patch("quarq.api.routes.report.get_risk_free_rate", return_value=0.03):
        resp = client.get(f"/report?portfolio_path={p}&format=json")

    assert resp.status_code == 200


def test_report_with_narrative(client) -> None:
    """POST /report?narrative=true includes narrative when LLM is mocked."""
    body = {
        "tickers": ["MC.PA"],
        "weights": [1.0],
        "start": "2024-01-01",
        "end": "2024-06-30",
        "benchmark": "^FCHI",
    }
    mock_prices = _mock_prices(["MC.PA", "^FCHI"])
    mock_llm = MagicMock()
    mock_llm.generate.return_value = "Portfolio delivered strong returns in the period."

    with patch("quarq.api.routes.report.fetch_portfolio", return_value=mock_prices), \
         patch("quarq.api.routes.report.get_risk_free_rate", return_value=0.03), \
         patch("quarq.llm.lmstudio.get_llm", return_value=mock_llm):
        resp = client.post("/report?format=html&narrative=true", json=body)

    assert resp.status_code == 200
    assert "Portfolio delivered strong returns" in resp.text
