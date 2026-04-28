"""AMF SFDR fund disclosure provider — downloads and parses the AMF Excel file."""

from __future__ import annotations

import io
import logging
from datetime import date

import pandas as pd
import requests

from quarq.exceptions import ProviderError
from quarq.ingest import cache
from quarq.ingest.base import BaseProvider

logger = logging.getLogger(__name__)

_AMF_SFDR_URL = (
    "https://www.amf-france.org/sites/institutionnel/files/2023-03/liste-opcvm-sfdr.xlsx"
)
_AMF_TTL = 86400  # 24 hours
_MIN_ROWS = 10

_KNOWN_SERIES = ("SFDR_FUNDS",)

# Known column name patterns for AMF SFDR Excel files (case-insensitive substring match)
_ISIN_COL_HINTS = ("isin", "code isin")
_ARTICLE_COL_HINTS = ("article", "sfdr", "classification")
_NAME_COL_HINTS = ("nom", "name", "denomination", "libelle")


class AMFProvider(BaseProvider):
    """Fetch AMF SFDR fund classification data from the AMF Excel download.

    No API key required. Downloads the publicly available Excel file from
    amf-france.org and parses fund SFDR article classifications.

    Attributes:
        fund_data: Full parsed DataFrame set after a successful fetch(), containing
            fund name, ISIN, and SFDR article columns.
    """

    fund_data: pd.DataFrame | None = None

    @property
    def name(self) -> str:
        """Provider identifier."""
        return "amf"

    def fetch(self, series_id: str, start: date, end: date) -> pd.DataFrame:
        """Download and parse the AMF SFDR Excel file.

        Sets self.fund_data with the full parsed content after a successful call.

        Args:
            series_id: Must be 'SFDR_FUNDS'.
            start: Inclusive start date (used for cache key; AMF file is a snapshot).
            end: Inclusive end date (used for cache key; AMF file is a snapshot).

        Returns:
            DataFrame with DatetimeIndex (publication date or today), value=1.0,
            series_id='SFDR_FUNDS', source='amf'.

        Raises:
            ProviderError: If series_id is unknown, the download fails, or the
                Excel file cannot be parsed.
        """
        if series_id not in _KNOWN_SERIES:
            raise ProviderError(
                f"Unknown AMF series '{series_id}'. Known: {list(_KNOWN_SERIES)}"
            )

        key = cache.make_key(self.name, series_id, start, end)
        cached = cache.get(key)
        if cached is not None:
            logger.debug("Cache hit for %s", key)
            return cached

        raw = _download_excel(_AMF_SFDR_URL)
        self.fund_data = _parse_excel(raw)

        today = pd.Timestamp.now().normalize()
        df = pd.DataFrame(
            {"value": [1.0], "series_id": ["SFDR_FUNDS"], "source": ["amf"]},
            index=pd.DatetimeIndex([today]),
        )

        cache.set(key, df, ttl_seconds=_AMF_TTL)
        return df

    pass  # helpers are module-level


def _download_excel(url: str) -> bytes:
    """Download an Excel file from the given URL.

    Args:
        url: URL of the Excel file to download.

    Returns:
        Raw bytes of the downloaded file.

    Raises:
        ProviderError: On connection error or non-200 HTTP status.
    """
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
    except requests.ConnectionError as exc:
        raise ProviderError(
            f"AMF SFDR download failed (connection error): {url}. "
            f"Check https://www.amf-france.org for the current file location."
        ) from exc
    except requests.HTTPError as exc:
        raise ProviderError(
            f"AMF SFDR download failed (HTTP {exc.response.status_code}): {url}. "
            f"Check https://www.amf-france.org for the current file location."
        ) from exc
    except requests.RequestException as exc:
        raise ProviderError(
            f"AMF SFDR download failed: {url} — {exc}. "
            f"Check https://www.amf-france.org for the current file location."
        ) from exc
    return resp.content


def _parse_excel(raw: bytes) -> pd.DataFrame:
    """Parse raw Excel bytes into a fund classification DataFrame.

    Args:
        raw: Raw bytes of the AMF SFDR Excel file.

    Returns:
        DataFrame with fund classification data.

    Raises:
        ProviderError: If openpyxl cannot parse the file.
    """
    try:
        df = pd.read_excel(io.BytesIO(raw), engine="openpyxl")
    except Exception as exc:
        raise ProviderError(
            f"AMF SFDR Excel parse failed: {exc}. "
            f"Check {_AMF_SFDR_URL} — the file format may have changed."
        ) from exc

    if len(df) < _MIN_ROWS:
        logger.warning(
            "AMF SFDR file has only %d rows (expected >= %d). "
            "The file format may have changed.",
            len(df),
            _MIN_ROWS,
        )

    return df


def _find_column(df: pd.DataFrame, hints: tuple[str, ...]) -> str | None:
    """Find the first column whose name matches any of the given hint substrings.

    Args:
        df: DataFrame to search.
        hints: Lowercase substrings to match against column names.

    Returns:
        Matching column name, or None if not found.
    """
    for col in df.columns:
        col_lower = str(col).lower()
        if any(hint in col_lower for hint in hints):
            return col
    return None


def _normalise_article(raw: str) -> str | None:
    """Normalise a raw SFDR article string to a canonical form.

    Args:
        raw: Raw article string from the Excel file (e.g. '8', 'Art. 8', 'Article 9').

    Returns:
        'Article 6', 'Article 8', or 'Article 9', or None if unrecognised.
    """
    raw_lower = raw.lower()
    for number in ("6", "8", "9"):
        if number in raw_lower:
            return f"Article {number}"
    return None


def get_sfdr_article(isin: str) -> str | None:
    """Look up the SFDR article classification for a given ISIN.

    Downloads and parses the AMF SFDR Excel file if needed.

    Args:
        isin: Fund ISIN code (e.g. 'FR0010527275').

    Returns:
        'Article 6', 'Article 8', or 'Article 9', or None if not found. Never raises.
    """
    provider = AMFProvider()
    try:
        provider.fetch("SFDR_FUNDS", date.today(), date.today())
    except Exception as exc:
        logger.warning("AMF SFDR fetch failed in get_sfdr_article: %s", exc)
        return None

    if provider.fund_data is None:
        return None

    isin_col = _find_column(provider.fund_data, _ISIN_COL_HINTS)
    article_col = _find_column(provider.fund_data, _ARTICLE_COL_HINTS)
    if isin_col is None or article_col is None:
        logger.warning("AMF fund_data missing expected ISIN or article columns")
        return None

    mask = (
        provider.fund_data[isin_col].astype(str).str.strip().str.upper()
        == isin.strip().upper()
    )
    matches = provider.fund_data.loc[mask, article_col]
    if matches.empty:
        return None

    return _normalise_article(str(matches.iloc[0]).strip())
