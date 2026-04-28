"""FRED REST API provider for macro rate series."""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd
import requests

from quarq.config import QuarqConfig
from quarq.exceptions import ProviderError
from quarq.ingest import cache
from quarq.ingest.base import BaseProvider

logger = logging.getLogger(__name__)

_FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"
_FRED_TTL = 86400  # 24 hours

# Internal mapping from quarq series IDs to FRED series IDs
_SERIES_MAP: dict[str, str] = {
    "OAT10Y": "IRLTLT01FRQ156N",
    "EUR_SPREAD": "BAMLHE00EHY0EY",
}


class FREDProvider(BaseProvider):
    """Fetch macro series from the FRED REST API.

    Requires a FRED API key stored in config.data.fred_api_key.
    Raises ProviderError immediately if the key is absent.
    """

    def __init__(self, cfg: QuarqConfig | None = None) -> None:
        """Initialise with config.

        Args:
            cfg: Loaded QuarqConfig instance. If None, loads from disk.
        """
        if cfg is None:
            from quarq.config import load_config

            cfg = load_config()
        self._cfg = cfg

    @property
    def name(self) -> str:
        """Provider identifier."""
        return "fred"

    def fetch(self, series_id: str, start: date, end: date) -> pd.DataFrame:
        """Fetch a FRED macro series.

        Args:
            series_id: quarq series identifier ('OAT10Y' or 'EUR_SPREAD').
            start: Inclusive start date.
            end: Inclusive end date.

        Returns:
            DataFrame with DatetimeIndex and columns value, series_id, source.

        Raises:
            ProviderError: If no API key, unknown series_id, or HTTP failure.
        """
        if not self._cfg.data.fred_api_key:
            logger.warning("FRED API key not set, cannot fetch %s", series_id)
            raise ProviderError("FRED API key required")

        fred_id = _SERIES_MAP.get(series_id)
        if fred_id is None:
            raise ProviderError(
                f"Unknown FRED series '{series_id}'. Known: {list(_SERIES_MAP)}"
            )

        key = cache.make_key(self.name, series_id, start, end)
        cached = cache.get(key)
        if cached is not None:
            logger.debug("Cache hit for %s", key)
            return cached

        params = {
            "series_id": fred_id,
            "observation_start": start.isoformat(),
            "observation_end": end.isoformat(),
            "api_key": self._cfg.data.fred_api_key,
            "file_type": "json",
        }
        try:
            resp = requests.get(_FRED_BASE, params=params, timeout=15)
            resp.raise_for_status()
            payload = resp.json()
        except requests.RequestException as exc:
            raise ProviderError(f"FRED request failed for {series_id}: {exc}") from exc
        except ValueError as exc:
            raise ProviderError(f"FRED response JSON invalid for {series_id}: {exc}") from exc

        observations = payload.get("observations", [])
        if not observations:
            raise ProviderError(f"FRED returned no observations for {series_id}")

        records = []
        for obs in observations:
            try:
                val = float(obs["value"])
                records.append({"date": obs["date"], "value": val})
            except (ValueError, KeyError):
                continue  # skip missing-value markers like "."

        if not records:
            raise ProviderError(f"FRED returned only missing values for {series_id}")

        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        df["series_id"] = series_id
        df["source"] = self.name
        df.index = pd.to_datetime(df.index)

        cache.set(key, df, ttl_seconds=_FRED_TTL)
        return df


def get_risk_free_rate(cfg: QuarqConfig) -> float:
    """Return the French 10Y OAT rate as a decimal, with config fallback.

    Args:
        cfg: Loaded QuarqConfig.

    Returns:
        Latest OAT10Y rate (e.g. 0.032) or cfg.portfolio.risk_free_rate_fallback.
        Never raises.
    """
    if not cfg.data.fred_api_key:
        logger.warning(
            "FRED API key not set; using risk_free_rate_fallback=%.4f",
            cfg.portfolio.risk_free_rate_fallback,
        )
        return cfg.portfolio.risk_free_rate_fallback

    from datetime import date, timedelta

    provider = FREDProvider(cfg)
    end = date.today()
    start = end - timedelta(days=90)
    try:
        df = provider.fetch("OAT10Y", start, end)
        latest = float(df["value"].iloc[-1])
        return latest / 100.0  # FRED returns percent, convert to decimal
    except Exception as exc:
        logger.warning("FRED OAT10Y fetch failed (%s); using fallback", exc)
        return cfg.portfolio.risk_free_rate_fallback
