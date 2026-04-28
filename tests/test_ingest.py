"""Tests for quarq ingest layer."""

from datetime import date
from pathlib import Path

import pandas as pd
import pytest


def test_base_provider_cannot_be_instantiated() -> None:
    """BaseProvider is abstract and raises TypeError on direct instantiation."""
    from quarq.ingest.base import BaseProvider

    with pytest.raises(TypeError):
        BaseProvider()  # type: ignore[abstract]


def test_cache_set_and_get_round_trips(tmp_path: Path) -> None:
    """Cache stores a DataFrame and returns it within TTL."""
    from quarq.ingest import cache

    cache.CACHE_DIR = tmp_path  # redirect to temp dir
    key = cache.make_key("equity", "MC.PA", date(2025, 1, 1), date(2025, 12, 31))
    df = pd.DataFrame(
        {"value": [100.0, 101.0], "series_id": ["MC.PA", "MC.PA"], "source": ["equity", "equity"]},
        index=pd.to_datetime(["2025-01-02", "2025-01-03"]),
    )
    cache.set(key, df, ttl_seconds=60)
    result = cache.get(key)
    assert result is not None
    pd.testing.assert_frame_equal(result, df)


def test_cache_expired_returns_none(tmp_path: Path) -> None:
    """Cache returns None for entries whose TTL has elapsed."""
    from quarq.ingest import cache

    cache.CACHE_DIR = tmp_path
    key = cache.make_key("equity", "MC.PA", date(2025, 1, 1), date(2025, 12, 31))
    df = pd.DataFrame(
        {"value": [100.0], "series_id": ["MC.PA"], "source": ["equity"]},
        index=pd.to_datetime(["2025-01-02"]),
    )
    cache.set(key, df, ttl_seconds=-1)  # already expired
    result = cache.get(key)
    assert result is None


def test_equity_provider_returns_correct_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    """EquityProvider.fetch returns DataFrame with correct schema."""
    from unittest.mock import patch

    from quarq.ingest import cache
    from quarq.ingest.equity import EquityProvider

    monkeypatch.setattr(cache, "get", lambda key: None)
    monkeypatch.setattr(cache, "set", lambda key, data, ttl_seconds: None)

    mock_close = pd.DataFrame(
        {"MC.PA": [150.0, 151.0, 152.0, 153.0, 154.0]},
        index=pd.to_datetime(
            ["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07", "2025-01-08"]
        ),
    )
    mock_close.index.name = "Date"

    with patch("yfinance.download", return_value=mock_close):
        provider = EquityProvider()
        df = provider.fetch("MC.PA", date(2025, 1, 1), date(2025, 1, 31))

    assert isinstance(df.index, pd.DatetimeIndex)
    assert list(df.columns) == ["value", "series_id", "source"]
    assert df["series_id"].iloc[0] == "MC.PA"
    assert df["source"].iloc[0] == "equity"
    assert df["value"].dtype == float


def test_equity_provider_raises_on_empty_data(monkeypatch: pytest.MonkeyPatch) -> None:
    """EquityProvider.fetch raises ProviderError when yfinance returns empty DataFrame."""
    from unittest.mock import patch

    from quarq.exceptions import ProviderError
    from quarq.ingest import cache
    from quarq.ingest.equity import EquityProvider

    monkeypatch.setattr(cache, "get", lambda key: None)
    monkeypatch.setattr(cache, "set", lambda key, data, ttl_seconds: None)

    empty = pd.DataFrame()
    with patch("yfinance.download", return_value=empty):
        provider = EquityProvider()
        with pytest.raises(ProviderError, match="No data returned"):
            provider.fetch("BADTICKER", date(2025, 1, 1), date(2025, 1, 31))


def test_fred_provider_raises_without_key() -> None:
    """FREDProvider.fetch raises ProviderError immediately when api key is empty."""
    from quarq.config import QuarqConfig
    from quarq.exceptions import ProviderError
    from quarq.ingest.fred import FREDProvider

    cfg = QuarqConfig()
    cfg.data.fred_api_key = ""
    provider = FREDProvider(cfg)
    with pytest.raises(ProviderError, match="FRED API key required"):
        provider.fetch("OAT10Y", date(2025, 1, 1), date(2025, 12, 31))


def test_fred_get_risk_free_rate_fallback() -> None:
    """get_risk_free_rate returns fallback float when no API key is configured."""
    from quarq.config import QuarqConfig
    from quarq.ingest.fred import get_risk_free_rate

    cfg = QuarqConfig()
    cfg.data.fred_api_key = ""
    cfg.portfolio.risk_free_rate_fallback = 0.03

    rate = get_risk_free_rate(cfg)
    assert rate == 0.03


def test_ecb_provider_parses_response(monkeypatch: pytest.MonkeyPatch) -> None:
    """ECBProvider.fetch parses a minimal valid ECB SDW JSON response."""
    import responses as rsps_lib

    from quarq.ingest import cache
    from quarq.ingest.ecb import ECBProvider

    monkeypatch.setattr(cache, "get", lambda key: None)
    monkeypatch.setattr(cache, "set", lambda key, data, ttl_seconds: None)

    ecb_url = "https://data-api.ecb.europa.eu/service/data/FM/B.U2.EUR.4F.KR.MRR_FR.LEV"
    mock_body = {
        "dataSets": [
            {
                "series": {
                    "0:0:0:0:0:0:0": {
                        "observations": {
                            "0": [4.5],
                            "1": [4.25],
                        }
                    }
                }
            }
        ],
        "structure": {
            "dimensions": {
                "observation": [
                    {
                        "values": [
                            {"id": "2024-06-12", "name": "2024-06-12"},
                            {"id": "2024-09-18", "name": "2024-09-18"},
                        ]
                    }
                ]
            }
        },
    }

    with rsps_lib.RequestsMock() as mock:
        mock.add(rsps_lib.GET, ecb_url, json=mock_body, status=200)
        provider = ECBProvider()
        df = provider.fetch("MRR_FR", date(2024, 1, 1), date(2024, 12, 31))

    assert isinstance(df.index, pd.DatetimeIndex)
    assert "value" in df.columns
    assert "series_id" in df.columns
    assert "source" in df.columns
    assert len(df) == 2


def test_ecb_provider_raises_on_bad_response(monkeypatch: pytest.MonkeyPatch) -> None:
    """ECBProvider.fetch raises ProviderError on HTTP 500."""
    import responses as rsps_lib

    from quarq.exceptions import ProviderError
    from quarq.ingest import cache
    from quarq.ingest.ecb import ECBProvider

    monkeypatch.setattr(cache, "get", lambda key: None)
    monkeypatch.setattr(cache, "set", lambda key, data, ttl_seconds: None)

    ecb_url = "https://data-api.ecb.europa.eu/service/data/FM/B.U2.EUR.4F.KR.MRR_FR.LEV"

    with rsps_lib.RequestsMock() as mock:
        mock.add(rsps_lib.GET, ecb_url, status=500)
        provider = ECBProvider()
        with pytest.raises(ProviderError):
            provider.fetch("MRR_FR", date(2024, 1, 1), date(2024, 12, 31))


def test_oecd_provider_raises_on_unknown_series() -> None:
    """OECDProvider.fetch raises ProviderError with informative message for unknown series."""
    from quarq.exceptions import ProviderError
    from quarq.ingest.oecd import OECDProvider

    provider = OECDProvider()
    with pytest.raises(ProviderError, match="Unknown OECD series"):
        provider.fetch("UNKNOWN_SERIES", date(2024, 1, 1), date(2024, 12, 31))


def test_provider_registry_get_provider() -> None:
    """get_provider returns correct instances; unknown name raises ProviderError."""
    from quarq.exceptions import ProviderError
    from quarq.ingest import PROVIDER_REGISTRY, get_provider
    from quarq.ingest.equity import EquityProvider

    assert "equity" in PROVIDER_REGISTRY
    assert "fred" in PROVIDER_REGISTRY
    assert "ecb" in PROVIDER_REGISTRY
    assert "oecd" in PROVIDER_REGISTRY

    provider = get_provider("equity")
    assert isinstance(provider, EquityProvider)

    with pytest.raises(ProviderError, match="Unknown provider"):
        get_provider("unknown")


def test_eurostat_provider_parses_response(monkeypatch: pytest.MonkeyPatch) -> None:
    """EurostatProvider.fetch parses a minimal valid Eurostat JSON response."""
    import responses as rsps_lib

    from quarq.ingest import cache
    from quarq.ingest.eurostat import EurostatProvider

    monkeypatch.setattr(cache, "get", lambda key: None)
    monkeypatch.setattr(cache, "set", lambda key, data, ttl_seconds: None)

    eurostat_url = (
        "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/prc_hicp_midx"
    )
    mock_body = {
        "value": {"0": 112.3, "1": 113.1},
        "dimension": {
            "time": {
                "category": {
                    "index": {"2023-01": 0, "2023-02": 1}
                }
            }
        },
    }

    with rsps_lib.RequestsMock() as mock:
        mock.add(rsps_lib.GET, eurostat_url, json=mock_body, status=200)
        provider = EurostatProvider()
        df = provider.fetch("HICP_FR", date(2023, 1, 1), date(2023, 12, 31))

    assert isinstance(df.index, pd.DatetimeIndex)
    assert list(df.columns) == ["value", "series_id", "source"]
    assert df["series_id"].iloc[0] == "HICP_FR"
    assert df["source"].iloc[0] == "eurostat"
    assert len(df) == 2


def test_eurostat_provider_raises_on_unknown_series() -> None:
    """EurostatProvider.fetch raises ProviderError for unrecognised series_id."""
    from quarq.exceptions import ProviderError
    from quarq.ingest.eurostat import EurostatProvider

    provider = EurostatProvider()
    with pytest.raises(ProviderError, match="Unknown Eurostat series"):
        provider.fetch("UNKNOWN_SERIES", date(2023, 1, 1), date(2023, 12, 31))


def test_eurostat_provider_raises_on_bad_response(monkeypatch: pytest.MonkeyPatch) -> None:
    """EurostatProvider.fetch raises ProviderError on HTTP 500."""
    import responses as rsps_lib

    from quarq.exceptions import ProviderError
    from quarq.ingest import cache
    from quarq.ingest.eurostat import EurostatProvider

    monkeypatch.setattr(cache, "get", lambda key: None)
    monkeypatch.setattr(cache, "set", lambda key, data, ttl_seconds: None)

    eurostat_url = (
        "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/prc_hicp_midx"
    )

    with rsps_lib.RequestsMock() as mock:
        mock.add(rsps_lib.GET, eurostat_url, status=500)
        provider = EurostatProvider()
        with pytest.raises(ProviderError):
            provider.fetch("HICP_FR", date(2023, 1, 1), date(2023, 12, 31))
