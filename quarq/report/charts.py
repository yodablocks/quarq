"""Plotly chart builders for quarq HTML reports.

Each function is pure: accepts DataFrames/dicts, returns go.Figure.
No plotly.express — uses graph_objects only (px wrappers broken in Plotly 6).
Color scheme: blues and greys. White background, gridlines off.
"""

from __future__ import annotations

import plotly.graph_objects as go
import pandas as pd


_BLUE = "#2563EB"
_BLUE_LIGHT = "#93C5FD"
_GREY = "#6B7280"
_BG = "white"

_LAYOUT_BASE = dict(
    paper_bgcolor=_BG,
    plot_bgcolor=_BG,
    font=dict(color="#111827", size=12),
    xaxis=dict(showgrid=False, zeroline=False),
    yaxis=dict(showgrid=False, zeroline=False),
    margin=dict(l=40, r=20, t=40, b=40),
)


def correlation_heatmap(returns: pd.DataFrame) -> go.Figure:
    """Render a correlation heatmap for a multi-ticker returns DataFrame.

    Args:
        returns: DataFrame where each column is a ticker's daily returns.

    Returns:
        go.Figure containing a Heatmap trace.
    """
    corr = returns.corr()
    labels = list(corr.columns)

    fig = go.Figure(
        go.Heatmap(
            z=corr.values.tolist(),
            x=labels,
            y=labels,
            colorscale=[[0, "#DBEAFE"], [0.5, _BLUE_LIGHT], [1, _BLUE]],
            zmin=-1,
            zmax=1,
            text=[[f"{v:.2f}" for v in row] for row in corr.values],
            texttemplate="%{text}",
        )
    )
    fig.update_layout(
        title="Correlation Matrix",
        **{k: v for k, v in _LAYOUT_BASE.items() if k not in ("xaxis", "yaxis")},
    )
    return fig


def weight_treemap(weights: dict[str, float], sleeve_map: dict[str, str]) -> go.Figure:
    """Render a treemap of portfolio weights grouped by sleeve.

    Args:
        weights: Mapping from ticker to flat weight (benchmark excluded).
        sleeve_map: Mapping from ticker to sleeve name.

    Returns:
        go.Figure containing a Treemap trace.
    """
    labels: list[str] = []
    parents: list[str] = []
    values: list[float] = []

    sleeves: set[str] = set(sleeve_map.get(t, "Equity") for t in weights)
    for sleeve in sleeves:
        labels.append(sleeve)
        parents.append("")
        values.append(0.0)

    for ticker, weight in weights.items():
        sleeve = sleeve_map.get(ticker, "Equity")
        labels.append(ticker)
        parents.append(sleeve)
        values.append(weight)

    fig = go.Figure(
        go.Treemap(
            labels=labels,
            parents=parents,
            values=values,
            marker=dict(colors=[_BLUE if p == "" else _BLUE_LIGHT for p in parents]),
            textinfo="label+percent entry",
        )
    )
    fig.update_layout(title="Portfolio Weights", paper_bgcolor=_BG, margin=dict(l=0, r=0, t=40, b=0))
    return fig


def cumulative_returns_line(
    port: pd.Series,
    bench: pd.Series,
    label: str = "Portfolio",
) -> go.Figure:
    """Render cumulative returns for portfolio and benchmark.

    Args:
        port: Daily portfolio returns Series.
        bench: Daily benchmark returns Series.
        label: Display label for the portfolio trace.

    Returns:
        go.Figure with two Scatter traces.
    """
    port_cum = (1 + port).cumprod()
    bench_cum = (1 + bench).cumprod()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=port_cum.index.tolist(),
            y=port_cum.values.tolist(),
            mode="lines",
            name=label,
            line=dict(color=_BLUE, width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=bench_cum.index.tolist(),
            y=bench_cum.values.tolist(),
            mode="lines",
            name="Benchmark",
            line=dict(color=_GREY, width=1, dash="dash"),
        )
    )
    fig.update_layout(title="Cumulative Returns", **_LAYOUT_BASE)
    return fig


def drawdown_area(returns: pd.Series) -> go.Figure:
    """Render a drawdown area chart.

    Args:
        returns: Daily portfolio returns Series.

    Returns:
        go.Figure with a filled Scatter trace (fill='tozeroy').
    """
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.cummax()
    dd = (cumulative - rolling_max) / rolling_max

    fig = go.Figure(
        go.Scatter(
            x=dd.index.tolist(),
            y=dd.values.tolist(),
            mode="lines",
            fill="tozeroy",
            name="Drawdown",
            line=dict(color=_BLUE, width=1),
            fillcolor="rgba(37,99,235,0.15)",
        )
    )
    fig.update_layout(title="Drawdown", **_LAYOUT_BASE)
    return fig


def rolling_sharpe(
    returns: pd.Series,
    rfr: float,
    window: int = 63,
) -> go.Figure:
    """Render a rolling Sharpe ratio chart.

    Args:
        returns: Daily portfolio returns Series.
        rfr: Annual risk-free rate as a decimal (e.g. 0.03).
        window: Rolling window in trading days (default 63 = ~1 quarter).

    Returns:
        go.Figure with a single Scatter trace.
    """
    daily_rf = rfr / 252
    excess = returns - daily_rf
    roll_mean = excess.rolling(window).mean()
    roll_std = returns.rolling(window).std()
    roll_sharpe = (roll_mean / roll_std) * (252 ** 0.5)

    fig = go.Figure(
        go.Scatter(
            x=roll_sharpe.index.tolist(),
            y=roll_sharpe.values.tolist(),
            mode="lines",
            name=f"Rolling Sharpe ({window}d)",
            line=dict(color=_BLUE, width=2),
        )
    )
    fig.update_layout(title=f"Rolling Sharpe ({window}-day)", **_LAYOUT_BASE)
    return fig
