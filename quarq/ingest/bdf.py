"""Banque de France webstat API provider for French-specific financial data."""

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

_BDF_BASE = "https://webstat.banque-france.fr/api/explore/v2.1/catalog/datasets"
_BDF_TTL = 86400  # 24 hours

# Maps quarq series IDs to (dataset_id, value_field) tuples.
# BdF API endpoints change occasionally — if a fetch fails with 404 the error
# message directs the user to check https://webstat.banque-france.fr for current IDs.
_SERIES_MAP: dict[str, tuple[str, str]] = {
    "OAT_10Y_FR": (
        "fm_vm_vm_fr_b2_vm_iuzta_hpe_fr0000131104_e",
        "value",
    ),
    "CREDIT_FR": (
        "bdf_credit_menages",
        "value",
    ),
}


class BDFProvider(BaseProvider):
    """Fetch French-specific financial data from the Banque de France webstat API.

    No API key required. Fails loudly on endpoint errors with actionable messages.
    Supports OAT_10Y_FR and CREDIT_FR.
    """

    @property
    def name(self) -> str:
        """Provider identifier."""
        return "bdf"

    def fetch(self, series_id: str, start: date, end: date) -> pd.DataFrame:
        """Fetch a French financial series from the Banque de France API.

        Args:
            series_id: quarq series identifier ('OAT_10Y_FR' or 'CREDIT_FR').
            start: Inclusive start date.
            end: Inclusive end date.

        Returns:
            DataFrame with DatetimeIndex and columns value, series_id, source.

        Raises:
            ProviderError: If series_id is unknown, the endpoint returns 404
                (with message suggesting BdF webstat portal for current dataset ID),
                or any other HTTP/parse error occurs.
        """
        if series_id not in _SERIES_MAP:
            raise ProviderError(
                f"Unknown BdF series '{series_id}'. Known: {list(_SERIES_MAP)}"
            )

        key = cache.make_key(self.name, series_id, start, end)
        cached = cache.get(key)
        if cached is not None:
            logger.debug("Cache hit for %s", key)
            return cached

        dataset_id, value_field = _SERIES_MAP[series_id]
        endpoint = f"{_BDF_BASE}/{dataset_id}/exports/json"
        params = {
            "where": f"date >= '{start.isoformat()}' and date <= '{end.isoformat()}'",
            "limit": 10000,
        }

        try:
            resp = requests.get(endpoint, params=params, timeout=30)
        except requests.RequestException as exc:
            raise ProviderError(f"BdF request failed for {series_id}: {exc}") from exc

        if resp.status_code == 404:
            raise ProviderError(
                f"BdF endpoint not found for {series_id} "
                f"(endpoint: {endpoint}). "
                f"The dataset ID '{dataset_id}' may have changed — "
                f"check https://webstat.banque-france.fr for the current dataset ID."
            )

        try:
            resp.raise_for_status()
            payload: list[dict[str, Any]] = resp.json()
        except requests.HTTPError as exc:
            raise ProviderError(
                f"BdF request failed for {series_id} (HTTP {exc.response.status_code}): "
                f"endpoint {endpoint}"
            ) from exc
        except ValueError as exc:
            raise ProviderError(f"BdF response JSON invalid for {series_id}: {exc}") from exc

        df = _parse_bdf_json(payload, series_id, self.name, value_field)
        cache.set(key, df, ttl_seconds=_BDF_TTL)
        return df


def _parse_bdf_json(
    payload: list[dict[str, Any]],
    series_id: str,
    source: str,
    value_field: str,
) -> pd.DataFrame:
    """Parse a BdF JSON export response into the standard DataFrame schema.

    Args:
        payload: List of record dicts from BdF API JSON export.
        series_id: quarq series identifier for metadata columns.
        source: Source name for the source column.
        value_field: Key in each record holding the numeric value.

    Returns:
        DataFrame with DatetimeIndex and columns value, series_id, source.

    Raises:
        ProviderError: If the payload is empty, malformed, or contains no usable rows.
    """
    if not payload:
        raise ProviderError(f"BdF returned empty response for {series_id}")

    records = []
    for row in payload:
        try:
            period = row["date"]
            val = float(row[value_field])
            records.append({"date": period, "value": val})
        except (KeyError, TypeError, ValueError):
            continue

    if not records:
        raise ProviderError(f"BdF returned no usable observations for {series_id}")

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df["series_id"] = series_id
    df["source"] = source
    return df


def get_oat_rate(start: date, end: date) -> pd.DataFrame | None:
    """Fetch French 10Y OAT yield from BdF as an alternative risk-free rate source.

    Args:
        start: Inclusive start date.
        end: Inclusive end date.

    Returns:
        DataFrame with standard schema, or None if the fetch fails. Never raises.
    """
    provider = BDFProvider()
    try:
        return provider.fetch("OAT_10Y_FR", start, end)
    except Exception as exc:
        logger.warning("BdF OAT fetch failed: %s", exc)
        return None
