"""Data provider fetch and listing endpoints."""

from __future__ import annotations

from datetime import date

from fastapi import APIRouter, HTTPException

from quarq.api.models import DataFetchResponse
from quarq.exceptions import ProviderError
from quarq.ingest import PROVIDER_REGISTRY, get_provider

router = APIRouter()


@router.get("/providers")
def get_providers() -> dict:
    """List all registered data providers.

    Returns:
        Dict with a single 'providers' key listing provider names.
    """
    return {"providers": list(PROVIDER_REGISTRY.keys())}


@router.get("/{provider}/{series_id}", response_model=DataFetchResponse)
def get_series(
    provider: str,
    series_id: str,
    start: date,
    end: date,
) -> DataFetchResponse:
    """Fetch a data series from a named provider.

    Args:
        provider: Provider name (must be in PROVIDER_REGISTRY).
        series_id: Series identifier (provider-specific).
        start: Inclusive start date (query param).
        end: Inclusive end date (query param).

    Returns:
        DataFetchResponse with series data as a list of date/value dicts.

    Raises:
        HTTPException 404: If provider is not in registry.
        HTTPException 503: If provider fetch fails.
    """
    if provider not in PROVIDER_REGISTRY:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown provider '{provider}'. Available: {list(PROVIDER_REGISTRY)}",
        )

    try:
        prov = get_provider(provider)
        df = prov.fetch(series_id, start, end)
    except ProviderError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    records = [
        {"date": idx.strftime("%Y-%m-%d"), "value": float(row["value"])}
        for idx, row in df.iterrows()
    ]

    return DataFetchResponse(
        series_id=series_id,
        provider=provider,
        start=start,
        end=end,
        records=len(records),
        data=records,
        cached=False,
    )
