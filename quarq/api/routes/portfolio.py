"""Portfolio metrics and provider listing endpoints."""

from __future__ import annotations

import time

from fastapi import APIRouter, HTTPException, Request

from quarq.api import metrics as m
from quarq.api.models import MetricsResponse, PortfolioRequest
from quarq.exceptions import ProviderError
from quarq.ingest import PROVIDER_REGISTRY
from quarq.ingest.equity import fetch_portfolio
from quarq.ingest.fred import get_risk_free_rate

router = APIRouter()


@router.post("/metrics", response_model=MetricsResponse)
def post_metrics(body: PortfolioRequest, request: Request) -> MetricsResponse:
    """Compute risk metrics for a portfolio.

    Args:
        body: PortfolioRequest with tickers, weights, date range, benchmark.
        request: FastAPI Request (provides app.state.config).

    Returns:
        MetricsResponse with Sharpe, drawdown, CAGR, VaR, beta, alpha.

    Raises:
        HTTPException 503: If equity or FRED data cannot be fetched.
    """
    cfg = request.app.state.config
    t0 = time.monotonic()

    try:
        prices = fetch_portfolio(
            tickers=body.tickers,
            start=body.start,
            end=body.end,
            benchmark=body.benchmark,
        )
    except ProviderError as exc:
        raise HTTPException(status_code=503, detail=f"Equity data fetch failed: {exc}")

    rfr = get_risk_free_rate(cfg)

    portfolio_returns = m.compute_portfolio_returns(prices, body.tickers, body.weights)

    bm_df = prices.get(body.benchmark)
    benchmark_returns = bm_df["value"].pct_change().dropna() if bm_df is not None else None

    beta_val = (
        m.beta(portfolio_returns, benchmark_returns) if benchmark_returns is not None else None
    )
    alpha_val = (
        m.alpha(portfolio_returns, benchmark_returns, beta_val, rfr)
        if benchmark_returns is not None
        else None
    )

    elapsed_ms = int((time.monotonic() - t0) * 1000)

    return MetricsResponse(
        tickers=body.tickers,
        weights=body.weights,
        start=body.start,
        end=body.end,
        benchmark=body.benchmark,
        sharpe_ratio=m.sharpe_ratio(portfolio_returns, rfr),
        max_drawdown=m.max_drawdown(portfolio_returns),
        cagr=m.cagr(portfolio_returns),
        volatility=m.volatility(portfolio_returns),
        var_95=m.var_95(portfolio_returns),
        beta=beta_val,
        alpha=alpha_val,
        risk_free_rate=rfr,
        computation_time_ms=elapsed_ms,
    )


@router.get("/providers")
def get_providers() -> dict:
    """List all registered data providers.

    Returns:
        Dict with a single 'providers' key listing provider names.
    """
    return {"providers": list(PROVIDER_REGISTRY.keys())}
