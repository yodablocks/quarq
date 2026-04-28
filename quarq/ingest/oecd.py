"""OECD API provider for French macro data (CPI, GDP)."""

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

_OECD_BASE = "https://stats.oecd.org/SDMX-JSON/data"
_OECD_TTL = 86400  # 24 hours

# Maps quarq series IDs to (dataset, filter_path) tuples
_SERIES_MAP: dict[str, tuple[str, str]] = {
    "CPI_FR": ("PRICES_CPI", "CPALTT01.FRA.GP.A"),
    "GDP_FR": ("QNA", "FRA.B1_GE.VPVOBARSA.Q"),
}


class OECDProvider(BaseProvider):
    """Fetch French macro series from the OECD SDMX-JSON API.

    No API key required.
    """

    @property
    def name(self) -> str:
        """Provider identifier."""
        return "oecd"

    def fetch(self, series_id: str, start: date, end: date) -> pd.DataFrame:
        """Fetch a French macro series from the OECD API.

        Args:
            series_id: quarq series identifier ('CPI_FR' or 'GDP_FR').
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
                f"Unknown OECD series '{series_id}'. Known: {list(_SERIES_MAP)}"
            )

        key = cache.make_key(self.name, series_id, start, end)
        cached = cache.get(key)
        if cached is not None:
            logger.debug("Cache hit for %s", key)
            return cached

        dataset, filter_path = _SERIES_MAP[series_id]
        url = f"{_OECD_BASE}/{dataset}/{filter_path}/all"
        params = {
            "startTime": start.strftime("%Y-%m"),
            "endTime": end.strftime("%Y-%m"),
            "dimensionAtObservation": "allDimensions",
        }

        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            payload: dict[str, Any] = resp.json()
        except requests.HTTPError as exc:
            raise ProviderError(
                f"OECD request failed for {series_id} (HTTP {exc.response.status_code})"
            ) from exc
        except requests.RequestException as exc:
            raise ProviderError(f"OECD request failed for {series_id}: {exc}") from exc
        except ValueError as exc:
            raise ProviderError(f"OECD response JSON invalid for {series_id}: {exc}") from exc

        df = _parse_sdmx_json(payload, series_id, self.name)
        cache.set(key, df, ttl_seconds=_OECD_TTL)
        return df


def _parse_sdmx_json(payload: dict[str, Any], series_id: str, source: str) -> pd.DataFrame:
    """Parse an OECD SDMX-JSON payload into the standard DataFrame schema.

    Args:
        payload: Parsed JSON dict from OECD API.
        series_id: quarq series identifier for metadata columns.
        source: Source name for the source column.

    Returns:
        DataFrame with DatetimeIndex and columns value, series_id, source.

    Raises:
        ProviderError: If the payload structure is unexpected or empty.
    """
    try:
        data_sets = payload["dataSets"]
        structure = payload["structure"]
        dimensions = structure["dimensions"]["observation"]

        # Find the time dimension (usually last)
        time_dim = next(
            (
                d
                for d in dimensions
                if d.get("role") == "time" or "TIME" in d.get("id", "").upper()
            ),
            dimensions[-1],
        )
        time_values = [v["id"] for v in time_dim["values"]]

        observations: dict[str, list] = data_sets[0]["observations"]
    except (KeyError, IndexError, StopIteration) as exc:
        raise ProviderError(f"OECD SDMX-JSON structure unexpected: {exc}") from exc

    records = []
    for key_str, obs_list in observations.items():
        try:
            # Key is colon-separated dimension indices; time index is last
            time_idx = int(key_str.split(":")[-1])
            period = time_values[time_idx]
            val = float(obs_list[0])
            records.append({"date": period, "value": val})
        except (ValueError, IndexError):
            continue

    if not records:
        raise ProviderError(f"OECD returned no usable observations for {series_id}")

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df["series_id"] = series_id
    df["source"] = source
    return df
