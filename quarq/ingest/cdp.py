"""CDP (Carbon Disclosure Project) emissions score provider via Socrata open data."""

from __future__ import annotations

import logging
from datetime import date
from io import StringIO

import pandas as pd
import requests

from quarq.exceptions import ProviderError
from quarq.ingest import cache
from quarq.ingest.base import BaseProvider

logger = logging.getLogger(__name__)

# CDP public Socrata endpoint — Global 500 Emissions and Response Status.
# The originally specified shjr-9ej4 resource was removed from the public portal.
# marp-zazk provides the same schema: company_name, country, reporting_year,
# performance_band (letter grade A/B/C/D/E/F), disclosure_score, scope emissions.
_CDP_SOCRATA_URL = "https://data.cdp.net/resource/marp-zazk.csv"
_CDP_TTL = 86400  # 24 hours

# Letter grade → numeric score mapping
_GRADE_MAP: dict[str, float] = {
    "A": 4.0,
    "A-": 3.7,
    "B": 3.0,
    "B-": 2.7,
    "C": 2.0,
    "D": 1.0,
    "E": 0.5,
    "F": 0.0,
}

_KNOWN_SERIES = ("CDP_SCORES",)

# Known column name patterns (case-insensitive substring match).
# marp-zazk columns: company_name, country, reporting_year, performance_band,
# disclosure_score, scope_1_metric_tonnes_co2e, scope_2_metric_tonnes_co2e
_COMPANY_COL_HINTS = ("company_name", "organization", "company", "account_name", "name")
_YEAR_COL_HINTS = ("reporting_year", "accounting_year", "year", "survey_year")
_SCORE_COL_HINTS = ("performance_band", "score", "rating", "grade", "response_score")
_COUNTRY_COL_HINTS = ("country", "country_region", "headquarters")


class CDPProvider(BaseProvider):
    """Fetch CDP climate disclosure scores for French companies via Socrata open data.

    No API key required, but the Socrata endpoint is rate-limited.
    Falls back to a ProviderError with manual download instructions if unavailable.
    """

    @property
    def name(self) -> str:
        """Provider identifier."""
        return "cdp"

    def fetch(self, series_id: str, start: date, end: date) -> pd.DataFrame:
        """Fetch CDP climate scores from the Socrata open data endpoint.

        Args:
            series_id: 'CDP_SCORES' to return all French companies, or a specific
                company name to filter to that company's data.
            start: Inclusive start date (filters by reporting year).
            end: Inclusive end date (filters by reporting year).

        Returns:
            DataFrame with DatetimeIndex (reporting year), value = Scope 1 emissions
            (metric tonnes CO2e) or numeric score if a scored dataset is available,
            series_id = company name, source = 'cdp'.

        Raises:
            ProviderError: If download fails, the endpoint is unavailable, or
                no data is found for the requested series_id.
        """
        key = cache.make_key(self.name, series_id, start, end)
        cached = cache.get(key)
        if cached is not None:
            logger.debug("Cache hit for %s", key)
            return cached

        raw_df = _download_csv()
        df = _build_dataframe(raw_df, series_id, start, end)

        cache.set(key, df, ttl_seconds=_CDP_TTL)
        return df


def _download_csv() -> pd.DataFrame:
    """Download the CDP Socrata CSV and return as a raw DataFrame.

    Returns:
        Raw DataFrame from the CSV endpoint.

    Raises:
        ProviderError: On connection error or non-200 HTTP status.
    """
    try:
        resp = requests.get(_CDP_SOCRATA_URL, timeout=60)
        resp.raise_for_status()
    except requests.ConnectionError as exc:
        raise ProviderError(
            f"CDP Socrata endpoint unavailable: {_CDP_SOCRATA_URL}. "
            f"Download the dataset manually from https://data.cdp.net and "
            f"place it at the path specified in your quarq config."
        ) from exc
    except requests.HTTPError as exc:
        raise ProviderError(
            f"CDP Socrata endpoint returned HTTP {exc.response.status_code}: "
            f"{_CDP_SOCRATA_URL}. "
            f"Download manually from https://data.cdp.net if the endpoint is down."
        ) from exc
    except requests.RequestException as exc:
        raise ProviderError(
            f"CDP request failed: {exc}. "
            f"Download manually from https://data.cdp.net."
        ) from exc

    try:
        return pd.read_csv(StringIO(resp.text))
    except Exception as exc:
        raise ProviderError(f"CDP CSV parse failed: {exc}") from exc


def _build_dataframe(
    raw: pd.DataFrame,
    series_id: str,
    start: date,
    end: date,
) -> pd.DataFrame:
    """Filter and transform the raw CDP CSV into standard schema.

    Args:
        raw: Raw DataFrame from the Socrata CSV endpoint.
        series_id: 'CDP_SCORES' for all French companies, or a company name to filter.
        start: Inclusive start year for filtering.
        end: Inclusive end year for filtering.

    Returns:
        DataFrame with DatetimeIndex, value (numeric score), series_id (company name),
        source = 'cdp'.

    Raises:
        ProviderError: If no data is found for the requested filters.
    """
    country_col = _find_column(raw, _COUNTRY_COL_HINTS)
    company_col = _find_column(raw, _COMPANY_COL_HINTS)
    year_col = _find_column(raw, _YEAR_COL_HINTS)
    score_col = _find_column(raw, _SCORE_COL_HINTS)

    if company_col is None or year_col is None or score_col is None:
        raise ProviderError(
            f"CDP CSV missing expected columns. Found: {list(raw.columns)}. "
            f"The Socrata endpoint schema may have changed."
        )

    df = raw.copy()

    # Filter to French companies when possible
    if country_col is not None:
        france_mask = df[country_col].astype(str).str.lower().str.contains("france", na=False)
        df = df[france_mask]

    # Filter by year range
    try:
        df[year_col] = pd.to_numeric(df[year_col], errors="coerce")
        df = df.dropna(subset=[year_col])
        df = df[
            (df[year_col] >= start.year) & (df[year_col] <= end.year)
        ]
    except Exception as exc:
        logger.warning("CDP year filtering failed: %s", exc)

    # Filter by company name if a specific company was requested
    if series_id != "CDP_SCORES":
        mask = df[company_col].astype(str).str.lower().str.contains(
            series_id.lower(), na=False
        )
        df = df[mask]
        if df.empty:
            raise ProviderError(
                f"CDP: no data found for company '{series_id}'. "
                f"Check the company name against the CDP dataset at https://data.cdp.net."
            )

    if df.empty:
        raise ProviderError(
            f"CDP: no French company data found for years {start.year}–{end.year}. "
            f"Try a wider date range or check https://data.cdp.net."
        )

    # Convert letter grades to numeric scores
    df = df.copy()
    df["_numeric_score"] = df[score_col].astype(str).str.strip().map(_GRADE_MAP)

    # Try direct numeric parse for rows where grade map returned NaN
    numeric_fallback = pd.to_numeric(df[score_col], errors="coerce")
    df["_numeric_score"] = df["_numeric_score"].combine_first(numeric_fallback)

    df = df.dropna(subset=["_numeric_score"])

    try:
        index = pd.to_datetime(df[year_col].astype(int).astype(str), format="%Y")
    except Exception:
        index = pd.DatetimeIndex([pd.Timestamp.now()] * len(df))

    result = pd.DataFrame(
        {
            "value": df["_numeric_score"].astype(float).values,
            "series_id": df[company_col].astype(str).values,
            "source": "cdp",
        },
        index=index,
    )
    result.index.name = None
    return result.sort_index()


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


def get_company_score(company_name: str) -> float | None:
    """Return the most recent numeric CDP score for a company.

    Args:
        company_name: Company name to look up (partial match, case-insensitive).

    Returns:
        Most recent numeric score (0.0–4.0), or None if not found. Never raises.
    """
    provider = CDPProvider()
    today = date.today()
    try:
        df = provider.fetch(company_name, date(today.year - 5, 1, 1), today)
        if df.empty:
            return None
        return float(df["value"].iloc[-1])
    except Exception as exc:
        logger.warning("CDP score lookup failed for '%s': %s", company_name, exc)
        return None
