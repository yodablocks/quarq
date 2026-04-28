"""Pure pandas/numpy portfolio metric computations.

No external finance libraries. All inputs are price DataFrames from EquityProvider.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_portfolio_returns(
    prices: dict[str, pd.DataFrame],
    tickers: list[str],
    weights: list[float],
) -> pd.Series:
    """Compute daily weighted portfolio returns.

    Args:
        prices: Dict mapping ticker -> DataFrame with 'value' column.
        tickers: Ordered list of ticker symbols matching weights.
        weights: Portfolio weights (must sum to 1.0).

    Returns:
        pd.Series of daily portfolio returns indexed by date.
    """
    close = pd.DataFrame({t: prices[t]["value"] for t in tickers if t in prices})
    returns = close.pct_change().dropna()
    w = np.array(weights[: len(returns.columns)])
    w = w / w.sum()
    portfolio_returns: pd.Series = returns.dot(w)
    return portfolio_returns


def sharpe_ratio(returns: pd.Series, risk_free_rate: float) -> float | None:
    """Compute annualised Sharpe ratio.

    Args:
        returns: Daily portfolio returns.
        risk_free_rate: Annual risk-free rate as a decimal (e.g. 0.03).

    Returns:
        Sharpe ratio, or None if volatility is zero.
    """
    if returns.empty:
        return None
    daily_rf = risk_free_rate / 252
    excess = returns - daily_rf
    vol = returns.std()
    if vol == 0:
        return None
    return float((excess.mean() / vol) * np.sqrt(252))


def max_drawdown(returns: pd.Series) -> float | None:
    """Compute maximum peak-to-trough drawdown.

    Args:
        returns: Daily portfolio returns.

    Returns:
        Maximum drawdown as a negative decimal (e.g. -0.25), or None if empty.
    """
    if returns.empty:
        return None
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    return float(drawdown.min())


def cagr(returns: pd.Series) -> float | None:
    """Compute compound annual growth rate.

    Args:
        returns: Daily portfolio returns.

    Returns:
        CAGR as a decimal (e.g. 0.12 = 12%), or None if fewer than 2 observations.
    """
    if len(returns) < 2:
        return None
    total = (1 + returns).prod()
    years = len(returns) / 252
    if years <= 0:
        return None
    return float(total ** (1 / years) - 1)


def volatility(returns: pd.Series) -> float | None:
    """Compute annualised volatility (standard deviation of returns).

    Args:
        returns: Daily portfolio returns.

    Returns:
        Annualised volatility as a decimal, or None if empty.
    """
    if returns.empty:
        return None
    return float(returns.std() * np.sqrt(252))


def var_95(returns: pd.Series) -> float | None:
    """Compute historical 95th-percentile Value at Risk.

    Args:
        returns: Daily portfolio returns.

    Returns:
        VaR as a negative decimal (e.g. -0.02 = 2% daily loss), or None if empty.
    """
    if returns.empty:
        return None
    return float(np.percentile(returns, 5))


def beta(portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float | None:
    """Compute portfolio beta relative to the benchmark.

    Args:
        portfolio_returns: Daily portfolio returns.
        benchmark_returns: Daily benchmark returns.

    Returns:
        Beta coefficient, or None if insufficient data or zero benchmark variance.
    """
    aligned = pd.DataFrame({"p": portfolio_returns, "b": benchmark_returns}).dropna()
    if len(aligned) < 2:
        return None
    bm_var = aligned["b"].var()
    if bm_var == 0:
        return None
    cov = aligned["p"].cov(aligned["b"])
    return float(cov / bm_var)


def alpha(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    beta_val: float | None,
    risk_free_rate: float,
) -> float | None:
    """Compute Jensen's alpha.

    Args:
        portfolio_returns: Daily portfolio returns.
        benchmark_returns: Daily benchmark returns.
        beta_val: Pre-computed beta (pass None to get None back).
        risk_free_rate: Annual risk-free rate as a decimal.

    Returns:
        Annualised Jensen's alpha as a decimal, or None if inputs are insufficient.
    """
    if beta_val is None or portfolio_returns.empty or benchmark_returns.empty:
        return None
    p_mean = portfolio_returns.mean() * 252
    b_mean = benchmark_returns.mean() * 252
    return float(p_mean - (risk_free_rate + beta_val * (b_mean - risk_free_rate)))
