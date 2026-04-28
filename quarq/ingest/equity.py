"""Equity price provider backed by yfinance."""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd
import yfinance as yf

from quarq.exceptions import ProviderError
from quarq.ingest import cache
from quarq.ingest.base import BaseProvider

logger = logging.getLogger(__name__)

_EQUITY_TTL = 3600  # 1 hour


class EquityProvider(BaseProvider):
    """Fetch adjusted closing prices via yfinance.

    Handles .PA tickers (Euronext Paris) and index symbols such as ^FCHI.
    """

    @property
    def name(self) -> str:
        """Provider identifier."""
        return "equity"

    def fetch(self, series_id: str, start: date, end: date) -> pd.DataFrame:
        """Fetch adjusted closing prices for a single ticker.

        Args:
            series_id: Ticker symbol, e.g. 'MC.PA' or '^FCHI'.
            start: Inclusive start date.
            end: Inclusive end date.

        Returns:
            DataFrame with DatetimeIndex and columns value, series_id, source.

        Raises:
            ProviderError: If yfinance returns empty data or raises.
        """
        key = cache.make_key(self.name, series_id, start, end)
        cached = cache.get(key)
        if cached is not None:
            logger.debug("Cache hit for %s", key)
            return cached

        try:
            raw: pd.DataFrame = yf.download(
                series_id, start=start, end=end, auto_adjust=True, progress=False
            )
        except Exception as exc:
            raise ProviderError(f"yfinance download failed for {series_id}: {exc}") from exc

        if raw.empty:
            raise ProviderError(f"No data returned by yfinance for {series_id}")

        # yfinance may return MultiIndex columns when a single ticker is passed
        if isinstance(raw.columns, pd.MultiIndex):
            close = raw["Close"][series_id]
        else:
            close = raw["Close"] if "Close" in raw.columns else raw.iloc[:, 0]

        if len(close) < 5:
            logger.warning(
                "Only %d rows returned for %s — possible bad ticker", len(close), series_id
            )

        df = pd.DataFrame(
            {
                "value": close.astype(float),
                "series_id": series_id,
                "source": self.name,
            },
            index=close.index,
        )
        df.index = pd.to_datetime(df.index)

        cache.set(key, df, ttl_seconds=_EQUITY_TTL)
        return df


def fetch_portfolio(
    tickers: list[str],
    start: date,
    end: date,
    benchmark: str = "^FCHI",
) -> dict[str, pd.DataFrame]:
    """Fetch a list of tickers plus a benchmark in one yfinance call.

    Args:
        tickers: List of ticker symbols.
        start: Inclusive start date.
        end: Inclusive end date.
        benchmark: Benchmark ticker appended to the download list.

    Returns:
        Dict mapping each ticker (and benchmark) to its standard DataFrame.

    Raises:
        ProviderError: If the benchmark fetch fails or all data is empty.
    """
    all_tickers = list(dict.fromkeys(tickers + [benchmark]))  # deduplicate, preserve order
    try:
        raw: pd.DataFrame = yf.download(
            all_tickers, start=start, end=end, auto_adjust=True, progress=False
        )
    except Exception as exc:
        raise ProviderError(f"yfinance portfolio download failed: {exc}") from exc

    if raw.empty:
        raise ProviderError("No data returned for portfolio tickers")

    provider = EquityProvider()
    result: dict[str, pd.DataFrame] = {}

    for ticker in all_tickers:
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                close = raw["Close"][ticker].dropna()
            else:
                close = raw["Close"].dropna()

            if close.empty:
                if ticker == benchmark:
                    raise ProviderError(f"Benchmark {benchmark} returned no data")
                continue

            df = pd.DataFrame(
                {
                    "value": close.astype(float),
                    "series_id": ticker,
                    "source": provider.name,
                },
                index=pd.to_datetime(close.index),
            )
            result[ticker] = df
        except ProviderError:
            raise
        except Exception as exc:
            if ticker == benchmark:
                raise ProviderError(f"Failed to process benchmark {benchmark}: {exc}") from exc
            raise ProviderError(f"Failed to process ticker {ticker}: {exc}") from exc

    if benchmark not in result:
        raise ProviderError(f"Benchmark {benchmark} could not be fetched")

    missing = [t for t in tickers if t not in result]
    if missing:
        raise ProviderError(
            f"No data returned for ticker(s): {missing}. "
            "Remove them from the portfolio or check that the symbols are valid."
        )

    return result
