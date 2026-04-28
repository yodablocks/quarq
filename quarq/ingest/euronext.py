"""Euronext public API provider for CAC 40 index data."""

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

_EURONEXT_BASE = "https://api.euronext.com/v1"
_EURONEXT_TTL = 3600  # 1 hour — index composition can change intraday

_CAC40_ISIN = "FR0003500008"
_CAC40_MIC = "XPAR"

_KNOWN_SERIES = ("CAC40_CONSTITUENTS", "CAC40_PRICE")


class EuronextProvider(BaseProvider):
    """Fetch CAC 40 index data from the Euronext public API.

    No API key required. Endpoints change occasionally; raises ProviderError
    with actionable messages on 404 rather than returning empty data.
    """

    @property
    def name(self) -> str:
        """Provider identifier."""
        return "euronext"

    def fetch(self, series_id: str, start: date, end: date) -> pd.DataFrame:
        """Fetch CAC 40 constituents or latest price from Euronext.

        Args:
            series_id: 'CAC40_CONSTITUENTS' or 'CAC40_PRICE'.
            start: Inclusive start date (ignored for snapshot series).
            end: Inclusive end date (ignored for snapshot series).

        Returns:
            DataFrame with DatetimeIndex and columns value, series_id, source.
            For CAC40_CONSTITUENTS: one row per constituent, value = weight.
            For CAC40_PRICE: single row, value = latest index price.

        Raises:
            ProviderError: If series_id is unknown, endpoint returns 404/5xx,
                or the response is malformed.
        """
        if series_id not in _KNOWN_SERIES:
            raise ProviderError(
                f"Unknown Euronext series '{series_id}'. Known: {list(_KNOWN_SERIES)}"
            )

        key = cache.make_key(self.name, series_id, start, end)
        cached = cache.get(key)
        if cached is not None:
            logger.debug("Cache hit for %s", key)
            return cached

        if series_id == "CAC40_CONSTITUENTS":
            df = self._fetch_constituents()
        else:
            df = self._fetch_price()

        cache.set(key, df, ttl_seconds=_EURONEXT_TTL)
        return df

    def _fetch_constituents(self) -> pd.DataFrame:
        """Fetch current CAC 40 constituent list and weights.

        Returns:
            DataFrame with one row per constituent, value = weight.

        Raises:
            ProviderError: On network error, 404, or malformed response.
        """
        endpoint = f"{_EURONEXT_BASE}/instruments/list"
        params = {"indexId": _CAC40_ISIN, "type": "Equity"}
        payload = _get_json(endpoint, params, "CAC40_CONSTITUENTS")
        return _parse_constituents(payload)

    def _fetch_price(self) -> pd.DataFrame:
        """Fetch latest CAC 40 index price.

        Returns:
            DataFrame with single row, value = index price.

        Raises:
            ProviderError: On network error, 404, or malformed response.
        """
        endpoint = f"{_EURONEXT_BASE}/trade/intraday"
        params = {"isin": _CAC40_ISIN, "mic": _CAC40_MIC}
        payload = _get_json(endpoint, params, "CAC40_PRICE")
        return _parse_price(payload)


def _get_json(endpoint: str, params: dict[str, str], series_id: str) -> Any:
    """Make a GET request and return parsed JSON.

    Args:
        endpoint: Full URL to request.
        params: Query parameters.
        series_id: Series identifier for error messages.

    Returns:
        Parsed JSON response body.

    Raises:
        ProviderError: On connection error, 404, other HTTP errors, or JSON parse failure.
    """
    try:
        resp = requests.get(endpoint, params=params, timeout=30)
    except requests.RequestException as exc:
        raise ProviderError(f"Euronext request failed for {series_id}: {exc}") from exc

    if resp.status_code == 404:
        raise ProviderError(
            f"Euronext endpoint not found for {series_id} (endpoint: {endpoint}). "
            f"Euronext API endpoints change occasionally — check https://api.euronext.com "
            f"for the current endpoint paths."
        )

    try:
        resp.raise_for_status()
    except requests.HTTPError as exc:
        raise ProviderError(
            f"Euronext request failed for {series_id} "
            f"(HTTP {exc.response.status_code}): {endpoint}"
        ) from exc

    try:
        return resp.json()
    except ValueError as exc:
        raise ProviderError(
            f"Euronext response JSON invalid for {series_id}: {exc}"
        ) from exc


def _parse_constituents(payload: Any) -> pd.DataFrame:
    """Parse CAC 40 constituent list JSON into standard DataFrame schema.

    Args:
        payload: Parsed JSON from Euronext instruments/list endpoint.

    Returns:
        DataFrame with one row per constituent, value = weight (float).

    Raises:
        ProviderError: If payload is empty or contains no usable rows.
    """
    instruments: list[dict[str, Any]] = []
    if isinstance(payload, dict):
        instruments = payload.get("instruments", payload.get("data", []))
    elif isinstance(payload, list):
        instruments = payload

    if not instruments:
        raise ProviderError("Euronext CAC40_CONSTITUENTS returned no instruments")

    records = []
    for item in instruments:
        try:
            ticker = str(item.get("ticker") or item.get("symbol") or item.get("isin", ""))
            weight = float(item.get("weight", item.get("indexWeight", 0.0)) or 0.0)
            records.append({"ticker": ticker, "value": weight})
        except (TypeError, ValueError):
            continue

    if not records:
        raise ProviderError("Euronext CAC40_CONSTITUENTS: no usable constituent rows in response")

    df = pd.DataFrame(records)
    df.index = pd.DatetimeIndex([pd.Timestamp.now()] * len(df))
    df["series_id"] = df["ticker"]
    df["source"] = "euronext"
    return df[["value", "series_id", "source"]]


def _parse_price(payload: Any) -> pd.DataFrame:
    """Parse CAC 40 intraday price JSON into standard DataFrame schema.

    Args:
        payload: Parsed JSON from Euronext trade/intraday endpoint.

    Returns:
        DataFrame with single row, value = latest index price.

    Raises:
        ProviderError: If payload is empty or price cannot be extracted.
    """
    price: float | None = None
    if isinstance(payload, dict):
        price = payload.get("lastPrice") or payload.get("price") or payload.get("value")
    elif isinstance(payload, list) and payload:
        first = payload[0]
        if isinstance(first, dict):
            price = first.get("lastPrice") or first.get("price")

    if price is None:
        raise ProviderError(
            "Euronext CAC40_PRICE: could not extract price from response. "
            "The endpoint response format may have changed."
        )

    df = pd.DataFrame(
        {"value": [float(price)], "series_id": ["CAC40_PRICE"], "source": ["euronext"]},
        index=pd.DatetimeIndex([pd.Timestamp.now()]),
    )
    return df


def get_cac40_constituents() -> pd.DataFrame | None:
    """Fetch the current CAC 40 constituent list and weights.

    Convenience wrapper that swallows all exceptions — used by the pipeline to
    validate portfolio tickers against the index.

    Returns:
        DataFrame with constituent data, or None if the fetch fails.
    """
    provider = EuronextProvider()
    today = date.today()
    try:
        return provider.fetch("CAC40_CONSTITUENTS", today, today)
    except Exception as exc:
        logger.warning("Euronext CAC40 constituents fetch failed: %s", exc)
        return None
