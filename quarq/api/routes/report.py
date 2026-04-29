"""GET /report and POST /report endpoints for HTML/JSON portfolio reports."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException, Query, Request, Response

from quarq.api import metrics as m
from quarq.api.models import MetricsResponse, PortfolioRequest
from quarq.exceptions import ConfigError, ProviderError
from quarq.ingest.equity import fetch_portfolio
from quarq.ingest.fred import get_risk_free_rate
from quarq.portfolio import PortfolioSpec, load_portfolio
from quarq.report.charts import (
    correlation_heatmap,
    cumulative_returns_line,
    drawdown_area,
    rolling_sharpe,
    weight_treemap,
)
from quarq.report.renderer import render_html

router = APIRouter()


def _compute_report(
    spec: PortfolioSpec,
    cfg: object,
    *,
    include_narrative: bool = False,
) -> tuple[MetricsResponse, dict, str | None]:
    """Fetch prices, compute metrics, build charts, optionally generate narrative.

    Args:
        spec: PortfolioSpec from TOML or PortfolioRequest.
        cfg: QuarqConfig loaded from app state.
        include_narrative: Whether to generate the LLM narrative paragraph.

    Returns:
        Tuple of (MetricsResponse, figures dict, narrative string or None).

    Raises:
        HTTPException 503: If equity data cannot be fetched.
    """
    try:
        prices = fetch_portfolio(
            tickers=spec.tickers,
            start=spec.start,
            end=spec.end,
            benchmark=spec.benchmark,
        )
    except ProviderError as exc:
        raise HTTPException(status_code=503, detail=f"Equity data fetch failed: {exc}")

    rfr = get_risk_free_rate(cfg)
    portfolio_returns = m.compute_portfolio_returns(prices, spec.tickers, spec.weights)

    bm_df = prices.get(spec.benchmark)
    benchmark_returns = bm_df["value"].pct_change().dropna() if bm_df is not None else None

    beta_val = (
        m.beta(portfolio_returns, benchmark_returns) if benchmark_returns is not None else None
    )
    alpha_val = (
        m.alpha(portfolio_returns, benchmark_returns, beta_val, rfr)
        if benchmark_returns is not None
        else None
    )

    metrics = MetricsResponse(
        tickers=spec.tickers,
        weights=spec.weights,
        start=spec.start,
        end=spec.end,
        benchmark=spec.benchmark,
        sharpe_ratio=m.sharpe_ratio(portfolio_returns, rfr),
        max_drawdown=m.max_drawdown(portfolio_returns),
        cagr=m.cagr(portfolio_returns),
        volatility=m.volatility(portfolio_returns),
        var_95=m.var_95(portfolio_returns),
        beta=beta_val,
        alpha=alpha_val,
        risk_free_rate=rfr,
    )

    # Build all 5 charts
    returns_df_parts = []
    for ticker in spec.tickers:
        df = prices.get(ticker)
        if df is not None:
            s = df["value"].pct_change().dropna().rename(ticker)
            returns_df_parts.append(s)
    if len(returns_df_parts) > 1:
        returns_df = pd.concat(returns_df_parts, axis=1).dropna()
    elif returns_df_parts:
        returns_df = returns_df_parts[0].to_frame()
    else:
        returns_df = pd.DataFrame()

    bench_returns_for_chart = benchmark_returns if benchmark_returns is not None else portfolio_returns

    figures = {
        "cumulative_returns": cumulative_returns_line(
            portfolio_returns, bench_returns_for_chart, label=spec.name
        ),
        "drawdown": drawdown_area(portfolio_returns),
        "rolling_sharpe": rolling_sharpe(portfolio_returns, rfr=rfr),
        "correlation_heatmap": correlation_heatmap(returns_df) if not returns_df.empty else None,
        "weight_treemap": weight_treemap(
            dict(zip(spec.tickers, spec.weights)), spec.sleeve_map
        ),
    }

    narrative: str | None = None
    if include_narrative:
        narrative = m.generate_narrative(metrics, cfg)

    return metrics, figures, narrative


def _render_response(
    spec: PortfolioSpec,
    metrics: MetricsResponse,
    figures: dict,
    narrative: str | None,
    fmt: str,
) -> Response:
    """Render the report as HTML or JSON.

    Args:
        spec: Portfolio specification.
        metrics: Computed metrics.
        figures: Chart figures dict.
        narrative: Optional narrative string.
        fmt: 'html' or 'json'.

    Returns:
        FastAPI Response (HTML or JSON).
    """
    report_date = date.today().isoformat()

    if fmt == "html":
        html = render_html(
            portfolio_name=spec.name,
            metrics=metrics,
            figures={k: v for k, v in figures.items() if v is not None},
            narrative=narrative,
            benchmark=spec.benchmark,
            period_start=spec.start.isoformat(),
            period_end=spec.end.isoformat(),
            currency=spec.currency,
            report_date=report_date,
        )
        return Response(content=html, media_type="text/html")

    # JSON format
    import plotly.io as pio

    chart_json = {
        k: pio.to_json(v) if v is not None else None
        for k, v in figures.items()
    }
    return Response(
        content=metrics.model_dump_json() + "\n" + str(chart_json),
        media_type="application/json",
    )


@router.get("")
def get_report(
    request: Request,
    portfolio_path: str = Query(..., description="Path to portfolio TOML file"),
    format: str = Query("html", description="Output format: html or json"),
    narrative: bool = Query(False, description="Include LLM narrative paragraph"),
) -> Response:
    """Generate an HTML or JSON report from a portfolio TOML file.

    Args:
        request: FastAPI Request (provides app.state.config).
        portfolio_path: Path to the portfolio TOML file.
        format: 'html' or 'json'.
        narrative: Whether to generate the LLM narrative.

    Returns:
        HTML Response or JSON Response.

    Raises:
        HTTPException 400: If the TOML file is invalid.
        HTTPException 503: If equity data cannot be fetched.
    """
    cfg = request.app.state.config
    try:
        spec = load_portfolio(Path(portfolio_path))
    except ConfigError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    metrics, figures, narr = _compute_report(spec, cfg, include_narrative=narrative)
    return _render_response(spec, metrics, figures, narr, fmt=format)


@router.post("")
def post_report(
    body: PortfolioRequest,
    request: Request,
    format: str = Query("html", description="Output format: html or json"),
    narrative: bool = Query(False, description="Include LLM narrative paragraph"),
) -> Response:
    """Generate an HTML or JSON report from an inline portfolio request.

    Args:
        body: PortfolioRequest with tickers, weights, dates, benchmark.
        request: FastAPI Request (provides app.state.config).
        format: 'html' or 'json'.
        narrative: Whether to generate the LLM narrative.

    Returns:
        HTML Response or JSON Response.

    Raises:
        HTTPException 503: If equity data cannot be fetched.
    """
    cfg = request.app.state.config

    spec = PortfolioSpec(
        name="Portfolio",
        benchmark=body.benchmark,
        start=body.start,
        end=body.end,
        currency=body.currency,
        tickers=body.tickers,
        weights=body.weights,
        sleeve_map={t: "Equity" for t in body.tickers},
    )

    inc_narr = narrative or body.include_narrative
    metrics, figures, narr = _compute_report(spec, cfg, include_narrative=inc_narr)
    return _render_response(spec, metrics, figures, narr, fmt=format)
