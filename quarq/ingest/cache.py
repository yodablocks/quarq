"""Pickle-based provider cache with per-entry TTL."""

from __future__ import annotations

import logging
import pickle
import time
from datetime import date
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

CACHE_DIR: Path = Path.home() / ".quarq" / "cache"


def make_key(provider: str, series_id: str, start: date, end: date) -> str:
    """Return a consistent cache key string.

    Args:
        provider: Provider name (e.g. 'equity').
        series_id: Series identifier.
        start: Start date.
        end: End date.

    Returns:
        Colon-separated key string.
    """
    return f"{provider}:{series_id}:{start.isoformat()}:{end.isoformat()}"


def _path_for(key: str) -> Path:
    """Map a cache key to a file path under CACHE_DIR."""
    safe = key.replace(":", "_").replace("/", "-").replace("^", "")
    return CACHE_DIR / f"{safe}.pkl"


def get(key: str) -> pd.DataFrame | None:
    """Return cached DataFrame if key exists and TTL has not expired.

    Args:
        key: Cache key from ``make_key``.

    Returns:
        DataFrame if cached and valid, otherwise None.
    """
    path = _path_for(key)
    if not path.exists():
        return None
    try:
        with path.open("rb") as f:
            entry: dict = pickle.load(f)
        if time.time() > entry["expires_at"]:
            logger.debug("Cache expired for key %s", key)
            return None
        return entry["data"]
    except Exception:
        logger.warning("Cache read failed for key %s; treating as miss", key)
        return None


def set(key: str, data: pd.DataFrame, ttl_seconds: int) -> None:
    """Write DataFrame to cache with expiry timestamp.

    Args:
        key: Cache key from ``make_key``.
        data: DataFrame to cache.
        ttl_seconds: Seconds until expiry (negative means already expired).
    """
    path = _path_for(key)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        entry = {"data": data, "expires_at": time.time() + ttl_seconds}
        with path.open("wb") as f:
            pickle.dump(entry, f)
    except Exception:
        logger.warning("Cache write failed for key %s; skipping", key)


def clear(provider: str | None = None) -> int:
    """Delete all cache entries, or only entries for a given provider.

    Args:
        provider: If given, delete only files whose names start with that provider.

    Returns:
        Count of deleted cache files.
    """
    if not CACHE_DIR.exists():
        return 0
    count = 0
    prefix = f"{provider}_" if provider else ""
    for pkl in CACHE_DIR.glob("*.pkl"):
        if not prefix or pkl.name.startswith(prefix):
            pkl.unlink()
            count += 1
    return count
