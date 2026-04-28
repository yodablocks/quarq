"""Eurostat REST API provider for harmonised French macro indicators."""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

import pandas as pd
import requests

from quarq.exceptions import ProviderError
from quarq.ingest import cache
from quarq.ingest.base import BaseProvider

logger = logging.getLogger(__name__)

_EUROSTAT_BASE = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data"
_EUROSTAT_TTL = 86400  # 24 hours

# Maps quarq series IDs to (dataset, filter_params) tuples
_SERIES_MAP: dict[str, tuple[str, dict[str, str]]] = {
    "HICP_FR": (
        "prc_hicp_midx",
        {"unit": "I15", "coicop": "CP00", "geo": "FR"},
    ),
    "UNEMP_FR": (
        "une_rt_m",
        {"age": "TOTAL", "sex": "T", "unit": "PC_ACT", "geo": "FR"},
    ),
    "GDP_FR_Q": (
        "namq_10_gdp",
        {"unit": "CLV10_MEUR", "na_item": "B1GQ", "geo": "FR"},
    ),
}


class EurostatProvider(BaseProvider):
    """Fetch harmonised French macro indicators from the Eurostat REST API.

    No API key required. Supports HICP_FR, UNEMP_FR, GDP_FR_Q.
    """

    @property
    def name(self) -> str:
        """Provider identifier."""
        return "eurostat"

    def fetch(self, series_id: str, start: date, end: date) -> pd.DataFrame:
        """Fetch a French macro series from the Eurostat REST API.

        Args:
            series_id: quarq series identifier ('HICP_FR', 'UNEMP_FR', 'GDP_FR_Q').
            start: Inclusive start date.
            end: Inclusive end date.

        Returns:
            DataFrame with DatetimeIndex and columns value, series_id, source.

        Raises:
            ProviderError: If series_id is unknown, endpoint is unreachable,
                or response cannot be parsed.
        """
        if series_id not in _SERIES_MAP:
            raise ProviderError(
                f"Unknown Eurostat series '{series_id}'. Known: {list(_SERIES_MAP)}"
            )

        key = cache.make_key(self.name, series_id, start, end)
        cached = cache.get(key)
        if cached is not None:
            logger.debug("Cache hit for %s", key)
            return cached

        dataset, filter_params = _SERIES_MAP[series_id]
        url = f"{_EUROSTAT_BASE}/{dataset}"
        params: dict[str, str] = {
            **filter_params,
            "startPeriod": start.strftime("%Y-%m"),
            "endPeriod": end.strftime("%Y-%m"),
            "format": "JSON",
        }

        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            payload: dict[str, Any] = resp.json()
        except requests.HTTPError as exc:
            raise ProviderError(
                f"Eurostat request failed for {series_id} (HTTP {exc.response.status_code})"
            ) from exc
        except requests.RequestException as exc:
            raise ProviderError(f"Eurostat request failed for {series_id}: {exc}") from exc
        except ValueError as exc:
            raise ProviderError(f"Eurostat response JSON invalid for {series_id}: {exc}") from exc

        df = _parse_eurostat_json(payload, series_id, self.name)
        cache.set(key, df, ttl_seconds=_EUROSTAT_TTL)
        return df


def _parse_eurostat_json(payload: dict[str, Any], series_id: str, source: str) -> pd.DataFrame:
    """Parse a Eurostat JSON response into the standard DataFrame schema.

    Args:
        payload: Parsed JSON dict from Eurostat API.
        series_id: quarq series identifier for metadata columns.
        source: Source name for the source column.

    Returns:
        DataFrame with DatetimeIndex and columns value, series_id, source.

    Raises:
        ProviderError: If the payload structure is unexpected or empty.
    """
    try:
        values: dict[str, float] = payload["value"]
        period_index: dict[str, int] = payload["dimension"]["time"]["category"]["index"]
    except (KeyError, TypeError) as exc:
        raise ProviderError(f"Eurostat JSON structure unexpected for {series_id}: {exc}") from exc

    # Invert period_index: position integer -> period string
    index_to_period: dict[int, str] = {v: k for k, v in period_index.items()}

    records = []
    for pos_str, value in values.items():
        try:
            pos = int(pos_str)
            period = index_to_period[pos]
            records.append({"date": period, "value": float(value)})
        except (ValueError, KeyError):
            continue

    if not records:
        raise ProviderError(f"Eurostat returned no usable observations for {series_id}")

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df["series_id"] = series_id
    df["source"] = source
    return df


def get_french_unemployment(start: date, end: date) -> pd.DataFrame | None:
    """Fetch French monthly unemployment rate from Eurostat. Never raises.

    Args:
        start: Inclusive start date.
        end: Inclusive end date.

    Returns:
        DataFrame with standard schema, or None if the fetch fails.
    """
    provider = EurostatProvider()
    try:
        return provider.fetch("UNEMP_FR", start, end)
    except Exception as exc:
        logger.warning("Eurostat unemployment fetch failed: %s", exc)
        return None
