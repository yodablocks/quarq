"""Portfolio TOML loader and PortfolioSpec dataclass."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from quarq.exceptions import ConfigError


@dataclass
class PortfolioSpec:
    """Flat representation of a portfolio loaded from a TOML file.

    Attributes:
        name: Portfolio display name.
        benchmark: Benchmark ticker symbol (e.g. '^FCHI').
        start: Analysis start date (inclusive).
        end: Analysis end date (inclusive).
        currency: Portfolio currency code (e.g. 'EUR').
        tickers: Flat ordered list of ticker symbols (benchmark excluded).
        weights: Flat weights parallel to tickers, summing to 1.0.
        sleeve_map: Mapping from ticker to sleeve name.
    """

    name: str
    benchmark: str
    start: date
    end: date
    currency: str
    tickers: list[str]
    weights: list[float]
    sleeve_map: dict[str, str]


def load_portfolio(path: Path) -> PortfolioSpec:
    """Load and validate a portfolio TOML file into a PortfolioSpec.

    Args:
        path: Path to the portfolio TOML file.

    Returns:
        Validated PortfolioSpec with flat tickers, weights, and sleeve_map.

    Raises:
        ConfigError: If the file cannot be read, is missing required fields,
            or if sleeve/holding weights do not sum to 1.0 (±0.01 tolerance).
    """
    try:
        with path.open("rb") as f:
            data = tomllib.load(f)
    except FileNotFoundError:
        raise ConfigError(f"Portfolio file not found: {path}")
    except Exception as exc:
        raise ConfigError(f"Failed to read portfolio file {path}: {exc}") from exc

    try:
        name = str(data["name"])
        benchmark = str(data["benchmark"])
        start = date.fromisoformat(str(data["start"]))
        end = date.fromisoformat(str(data["end"]))
        currency = str(data.get("currency", "EUR"))
    except KeyError as exc:
        raise ConfigError(f"Portfolio TOML missing required field: {exc}") from exc
    except ValueError as exc:
        raise ConfigError(f"Portfolio TOML invalid date format: {exc}") from exc

    sleeves = data.get("sleeve", [])

    tickers: list[str] = []
    weights: list[float] = []
    sleeve_map: dict[str, str] = {}

    if sleeves:
        sleeve_weights = [float(s["weight"]) for s in sleeves]
        _validate_sum(sleeve_weights, "Global sleeve weights")

        for sleeve in sleeves:
            sleeve_name = str(sleeve["name"])
            sleeve_weight = float(sleeve["weight"])
            holdings = sleeve.get("holding", [])

            holding_weights = [float(h["weight"]) for h in holdings]
            _validate_sum(holding_weights, f"Sleeve '{sleeve_name}' holding weights")

            for holding in holdings:
                ticker = str(holding["ticker"])
                flat_weight = sleeve_weight * float(holding["weight"])
                tickers.append(ticker)
                weights.append(flat_weight)
                sleeve_map[ticker] = sleeve_name
    else:
        flat_holdings = data.get("holding", [])
        if not flat_holdings:
            raise ConfigError(
                "Portfolio TOML must contain either [[sleeve]] sections or [[holding]] entries."
            )
        holding_weights = [float(h["weight"]) for h in flat_holdings]
        _validate_sum(holding_weights, "Holding weights")
        for holding in flat_holdings:
            ticker = str(holding["ticker"])
            tickers.append(ticker)
            weights.append(float(holding["weight"]))
            sleeve_map[ticker] = "Equity"

    return PortfolioSpec(
        name=name,
        benchmark=benchmark,
        start=start,
        end=end,
        currency=currency,
        tickers=tickers,
        weights=weights,
        sleeve_map=sleeve_map,
    )


def _validate_sum(values: list[float], label: str, tolerance: float = 0.01) -> None:
    """Raise ConfigError if values do not sum to 1.0 within tolerance.

    Args:
        values: List of weights to validate.
        label: Human-readable label for error messages.
        tolerance: Acceptable deviation from 1.0.

    Raises:
        ConfigError: If abs(sum(values) - 1.0) > tolerance.
    """
    if not values:
        return
    total = sum(values)
    if abs(total - 1.0) > tolerance:
        raise ConfigError(
            f"{label} must sum to 1.0 (±{tolerance}), got {total:.4f}."
        )
