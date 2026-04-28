"""Abstract base class for all quarq data providers."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date

import pandas as pd

from quarq.exceptions import ProviderError  # noqa: F401 — re-exported for convenience

logger = logging.getLogger(__name__)


class BaseProvider(ABC):
    """Contract every data provider must implement.

    Subclasses must implement ``fetch`` and the ``name`` property.
    Returned DataFrames must have a DatetimeIndex and columns:
    ``value`` (float64), ``series_id`` (str), ``source`` (str).
    """

    @abstractmethod
    def fetch(self, series_id: str, start: date, end: date) -> pd.DataFrame:
        """Fetch a time series and return it in standard schema.

        Args:
            series_id: Provider-specific identifier (ticker, FRED code, etc.).
            start: Inclusive start date.
            end: Inclusive end date.

        Returns:
            DataFrame with DatetimeIndex and columns value, series_id, source.

        Raises:
            ProviderError: On any fetch failure. Never returns partial data.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Short lowercase identifier for this provider (e.g. 'equity')."""
        ...


@dataclass
class ProviderResult:
    """Metadata wrapper around a fetched DataFrame.

    Attributes:
        series_id: The identifier that was fetched.
        source: Provider name.
        start: Requested start date.
        end: Requested end date.
        data: The fetched DataFrame.
        cached: True if data came from cache.
    """

    series_id: str
    source: str
    start: date
    end: date
    data: pd.DataFrame
    cached: bool
