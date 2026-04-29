"""Tests for quarq/portfolio.py — PortfolioSpec loader."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from quarq.exceptions import ConfigError
from quarq.portfolio import PortfolioSpec, load_portfolio


_VALID_TOML = """\
name = "Test Portfolio"
benchmark = "^FCHI"
start = "2024-01-01"
end = "2024-12-31"
currency = "EUR"

[[sleeve]]
name = "Growth"
weight = 0.60

[[sleeve.holding]]
ticker = "MC.PA"
weight = 0.50

[[sleeve.holding]]
ticker = "AIR.PA"
weight = 0.50

[[sleeve]]
name = "Defensive"
weight = 0.40

[[sleeve.holding]]
ticker = "SAN.PA"
weight = 1.0
"""


def _write_toml(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "portfolio.toml"
    p.write_text(content, encoding="utf-8")
    return p


def test_load_valid_portfolio(tmp_path: Path) -> None:
    """Valid two-sleeve TOML returns correct PortfolioSpec."""
    path = _write_toml(tmp_path, _VALID_TOML)
    spec = load_portfolio(path)

    assert isinstance(spec, PortfolioSpec)
    assert spec.name == "Test Portfolio"
    assert spec.benchmark == "^FCHI"
    assert spec.start == date(2024, 1, 1)
    assert spec.end == date(2024, 12, 31)
    assert spec.currency == "EUR"
    assert set(spec.tickers) == {"MC.PA", "AIR.PA", "SAN.PA"}
    assert abs(sum(spec.weights) - 1.0) < 0.01


def test_flat_weights_correct(tmp_path: Path) -> None:
    """Flat weights are sleeve_weight × holding_weight."""
    path = _write_toml(tmp_path, _VALID_TOML)
    spec = load_portfolio(path)

    idx = spec.tickers.index("MC.PA")
    assert abs(spec.weights[idx] - 0.30) < 0.001  # 0.60 * 0.50

    idx2 = spec.tickers.index("SAN.PA")
    assert abs(spec.weights[idx2] - 0.40) < 0.001  # 0.40 * 1.0


def test_sleeve_map_populated(tmp_path: Path) -> None:
    """sleeve_map maps each ticker to its sleeve name."""
    path = _write_toml(tmp_path, _VALID_TOML)
    spec = load_portfolio(path)

    assert spec.sleeve_map["MC.PA"] == "Growth"
    assert spec.sleeve_map["AIR.PA"] == "Growth"
    assert spec.sleeve_map["SAN.PA"] == "Defensive"


def test_holding_weights_not_summing_to_one_raises(tmp_path: Path) -> None:
    """Holding weights that do not sum to 1.0 raise ConfigError."""
    bad = _VALID_TOML.replace('weight = 0.50\n\n[[sleeve.holding]]\nticker = "AIR.PA"\nweight = 0.50', 'weight = 0.60\n\n[[sleeve.holding]]\nticker = "AIR.PA"\nweight = 0.60')
    path = _write_toml(tmp_path, bad)
    with pytest.raises(ConfigError, match="holding weights"):
        load_portfolio(path)


def test_sleeve_weights_not_summing_to_one_raises(tmp_path: Path) -> None:
    """Global sleeve weights that do not sum to 1.0 raise ConfigError."""
    bad = _VALID_TOML.replace("weight = 0.60\n\n[[sleeve.holding]]", "weight = 0.70\n\n[[sleeve.holding]]")
    path = _write_toml(tmp_path, bad)
    with pytest.raises(ConfigError, match="sleeve weights"):
        load_portfolio(path)


def test_missing_file_raises(tmp_path: Path) -> None:
    """Loading a non-existent file raises ConfigError."""
    with pytest.raises(ConfigError, match="not found"):
        load_portfolio(tmp_path / "missing.toml")
