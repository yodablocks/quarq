"""Tests for quarq ingest layer."""

from datetime import date
from pathlib import Path

import pandas as pd
import pytest


def test_base_provider_cannot_be_instantiated() -> None:
    """BaseProvider is abstract and raises TypeError on direct instantiation."""
    from quarq.ingest.base import BaseProvider

    with pytest.raises(TypeError):
        BaseProvider()  # type: ignore[abstract]


def test_cache_set_and_get_round_trips(tmp_path: Path) -> None:
    """Cache stores a DataFrame and returns it within TTL."""
    from quarq.ingest import cache

    cache.CACHE_DIR = tmp_path  # redirect to temp dir
    key = cache.make_key("equity", "MC.PA", date(2025, 1, 1), date(2025, 12, 31))
    df = pd.DataFrame(
        {"value": [100.0, 101.0], "series_id": ["MC.PA", "MC.PA"], "source": ["equity", "equity"]},
        index=pd.to_datetime(["2025-01-02", "2025-01-03"]),
    )
    cache.set(key, df, ttl_seconds=60)
    result = cache.get(key)
    assert result is not None
    pd.testing.assert_frame_equal(result, df)


def test_cache_expired_returns_none(tmp_path: Path) -> None:
    """Cache returns None for entries whose TTL has elapsed."""
    from quarq.ingest import cache

    cache.CACHE_DIR = tmp_path
    key = cache.make_key("equity", "MC.PA", date(2025, 1, 1), date(2025, 12, 31))
    df = pd.DataFrame(
        {"value": [100.0], "series_id": ["MC.PA"], "source": ["equity"]},
        index=pd.to_datetime(["2025-01-02"]),
    )
    cache.set(key, df, ttl_seconds=-1)  # already expired
    result = cache.get(key)
    assert result is None
