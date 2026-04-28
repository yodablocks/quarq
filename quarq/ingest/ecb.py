"""ECB Statistical Data Warehouse REST provider."""

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

_ECB_URL = "https://data-api.ecb.europa.eu/service/data/FM/B.U2.EUR.4F.KR.MRR_FR.LEV"
_ECB_TTL = 86400  # 24 hours


class ECBProvider(BaseProvider):
    """Fetch ECB main refinancing rate from the ECB SDW REST API.

    No API key required. The series_id parameter is accepted for interface
    compatibility but ignored in v1 (single-series endpoint).
    """

    @property
    def name(self) -> str:
        """Provider identifier."""
        return "ecb"

    def fetch(self, series_id: str, start: date, end: date) -> pd.DataFrame:
        """Fetch the ECB main refinancing rate time series.

        Args:
            series_id: Accepted for interface compatibility; ignored in v1.
            start: Inclusive start date (used as cache key; endpoint returns full history).
            end: Inclusive end date.

        Returns:
            DataFrame with DatetimeIndex and columns value, series_id, source.

        Raises:
            ProviderError: On HTTP error or malformed JSON.
        """
        key = cache.make_key(self.name, series_id, start, end)
        cached = cache.get(key)
        if cached is not None:
            logger.debug("Cache hit for %s", key)
            return cached

        try:
            resp = requests.get(
                _ECB_URL,
                headers={"Accept": "application/json"},
                timeout=15,
            )
            resp.raise_for_status()
            payload = resp.json()
        except requests.HTTPError as exc:
            raise ProviderError(
                f"ECB SDW request failed (HTTP {exc.response.status_code})"
            ) from exc
        except requests.RequestException as exc:
            raise ProviderError(f"ECB SDW request failed: {exc}") from exc
        except ValueError as exc:
            raise ProviderError(f"ECB SDW response JSON invalid: {exc}") from exc

        try:
            obs_map: dict[str, list] = (
                payload["dataSets"][0]["series"]["0:0:0:0:0:0:0"]["observations"]
            )
            date_values: list[dict] = (
                payload["structure"]["dimensions"]["observation"][0]["values"]
            )
        except (KeyError, IndexError) as exc:
            raise ProviderError(f"ECB SDW JSON structure unexpected: {exc}") from exc

        records = []
        for idx_str, obs_list in obs_map.items():
            try:
                period = date_values[int(idx_str)]["id"]
                val = float(obs_list[0])
                records.append({"date": period, "value": val})
            except (ValueError, IndexError, KeyError):
                continue

        if not records:
            raise ProviderError("ECB SDW returned no usable observations")

        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        df["series_id"] = series_id
        df["source"] = self.name

        cache.set(key, df, ttl_seconds=_ECB_TTL)
        return df


def get_ecb_rate(cfg: QuarqConfig) -> float | None:  # noqa: ARG001
    """Return the latest ECB policy rate as a decimal, or None if unreachable.

    Args:
        cfg: Loaded QuarqConfig (reserved for future per-config behaviour).

    Returns:
        Latest rate divided by 100 (e.g. 0.045 for 4.5%), or None on failure.
        Never raises.
    """
    from datetime import date, timedelta

    provider = ECBProvider()
    end = date.today()
    start = end - timedelta(days=365)
    try:
        df = provider.fetch("MRR_FR", start, end)
        return float(df["value"].iloc[-1]) / 100.0
    except Exception as exc:
        logger.warning("ECB rate fetch failed: %s", exc)
        return None
