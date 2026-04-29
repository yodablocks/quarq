"""Numerical correctness tests for quarq/api/metrics.py."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from quarq.api.metrics import (
    alpha,
    beta,
    cagr,
    max_drawdown,
    sharpe_ratio,
    var_95,
    volatility,
)


def _make_returns(values: list[float], start: str = "2024-01-02") -> pd.Series:
    index = pd.date_range(start=start, periods=len(values), freq="B")
    return pd.Series(values, index=index)


def test_sharpe_constant_positive_return() -> None:
    """Sharpe with constant positive return above rfr should be large and positive."""
    returns = _make_returns([0.001] * 252)
    result = sharpe_ratio(returns, risk_free_rate=0.0)
    assert result is not None
    assert result > 5.0


def test_sharpe_zero_volatility_returns_none() -> None:
    """Sharpe with exactly-zero volatility returns None."""
    returns = _make_returns([0.0] * 100)
    result = sharpe_ratio(returns, risk_free_rate=0.0)
    assert result is None


def test_max_drawdown_monotone_decline() -> None:
    """Drawdown of -10% -10% -10%: cumulative peak is 0.9, trough is 0.729.
    Max drawdown = (0.729 - 0.9) / 0.9 ≈ -0.19."""
    returns = _make_returns([-0.10, -0.10, -0.10])
    result = max_drawdown(returns)
    assert result is not None
    assert abs(result - (-0.19)) < 0.001


def test_max_drawdown_positive_only_is_zero() -> None:
    """Portfolio that only rises has drawdown of 0."""
    returns = _make_returns([0.01, 0.02, 0.005, 0.01])
    result = max_drawdown(returns)
    assert result is not None
    assert result == pytest.approx(0.0, abs=1e-9)


def test_cagr_one_year_known_return() -> None:
    """252 business days span ~351 calendar days; CAGR is computed over actual calendar days."""
    daily_r = (1.10 ** (1 / 252)) - 1
    returns = _make_returns([daily_r] * 252)
    result = cagr(returns)
    assert result is not None
    # 351 calendar days → (1.10^(252/252))^(365.25/351) ≈ 1.0427, so CAGR ≈ 0.104
    assert abs(result - 0.104) < 0.002


def test_beta_uncorrelated_returns_near_zero() -> None:
    """Portfolio returns uncorrelated with benchmark should have beta ≈ 0."""
    rng = np.random.default_rng(42)
    port = _make_returns(rng.normal(0, 0.01, 252).tolist())
    bench = _make_returns(rng.normal(0, 0.01, 252).tolist())
    result = beta(port, bench)
    assert result is not None
    assert abs(result) < 0.15


def test_alpha_benchmark_clone_is_zero() -> None:
    """Portfolio identical to benchmark should have alpha ≈ 0."""
    rng = np.random.default_rng(7)
    bench_vals = rng.normal(0.0004, 0.01, 252).tolist()
    port = _make_returns(bench_vals)
    bench = _make_returns(bench_vals)
    beta_val = beta(port, bench)
    result = alpha(port, bench, beta_val, risk_free_rate=0.03)
    assert result is not None
    assert abs(result) < 0.01
