"""HTML and PDF report renderer for quarq.

render_html: Jinja2 template + embedded Plotly JS (no CDN).
render_pdf: Playwright-based PDF export (requires quarq[full]).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import plotly.graph_objects as go
import plotly.io as pio
from jinja2 import Environment, FileSystemLoader
from markupsafe import Markup

from quarq import __version__
from quarq.exceptions import QuarqError

_TEMPLATE_DIR = Path(__file__).parent
_TEMPLATE_NAME = "template.html"


def _pct(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.1f}%"


def _round2(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.2f}"


def _embed_figure(fig: go.Figure, first: bool) -> str:
    """Return HTML div for a Plotly figure.

    Args:
        fig: Plotly Figure to embed.
        first: If True, embed Plotly JS inline; subsequent figures reference it.

    Returns:
        HTML string with embedded chart.
    """
    include_js: str | bool = "inline" if first else False
    return pio.to_html(
        fig,
        full_html=False,
        include_plotlyjs=include_js,
        config={"displayModeBar": False},
    )


def render_html(
    portfolio_name: str,
    metrics: Any,
    figures: dict[str, go.Figure],
    template_path: Path | None = None,
    narrative: str | None = None,
    benchmark: str = "",
    period_start: str = "",
    period_end: str = "",
    currency: str = "EUR",
    report_date: str = "",
) -> str:
    """Render a complete HTML report from metrics and Plotly figures.

    Args:
        portfolio_name: Display name of the portfolio.
        metrics: MetricsResponse or any object with CAGR, Sharpe etc. attributes.
        figures: Dict mapping chart names to go.Figure instances. Expected keys:
            'cumulative_returns', 'drawdown', 'rolling_sharpe',
            'correlation_heatmap', 'weight_treemap'.
        template_path: Optional path to a custom Jinja2 template directory.
        narrative: Optional narrative paragraph; hidden if empty or None.
        benchmark: Benchmark ticker label.
        period_start: Analysis start date string.
        period_end: Analysis end date string.
        currency: Currency code.
        report_date: Report generation date string.

    Returns:
        Rendered HTML as a string.
    """
    tmpl_dir = template_path or _TEMPLATE_DIR
    env = Environment(
        loader=FileSystemLoader(str(tmpl_dir)),
        autoescape=True,
    )
    env.filters["pct"] = _pct
    env.filters["round2"] = _round2

    ordered_keys = [
        "cumulative_returns",
        "drawdown",
        "rolling_sharpe",
        "correlation_heatmap",
        "weight_treemap",
    ]
    embedded: dict[str, Markup] = {}
    first = True
    for key in ordered_keys:
        fig = figures.get(key)
        if fig is not None:
            embedded[key] = Markup(_embed_figure(fig, first=first))
            first = False
        else:
            embedded[key] = Markup("")

    template = env.get_template(_TEMPLATE_NAME)
    return template.render(
        portfolio_name=portfolio_name,
        metrics=metrics,
        charts=embedded,
        narrative=narrative or "",
        benchmark=benchmark,
        period_start=period_start,
        period_end=period_end,
        currency=currency,
        report_date=report_date,
        version=__version__,
    )


def render_pdf(html: str, output_path: Path) -> Path:
    """Render an HTML string to PDF using Playwright.

    Args:
        html: Complete HTML document string.
        output_path: Destination path for the PDF file.

    Returns:
        The output_path after writing.

    Raises:
        QuarqError: If Playwright is not installed or PDF generation fails.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise QuarqError(
            "PDF export requires Playwright. Install with: pip install 'quarq[full]'"
        ) from exc

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            page.set_content(html, wait_until="networkidle")
            page.pdf(path=str(output_path), format="A4", print_background=True)
            browser.close()
    except Exception as exc:
        raise QuarqError(f"PDF generation failed: {exc}") from exc

    return output_path
