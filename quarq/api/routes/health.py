"""Health and status endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Request

from quarq.api.models import HealthResponse
from quarq.status import StatusResult, run_status_checks

router = APIRouter()


def _status_to_response(result: StatusResult) -> HealthResponse:
    """Convert a StatusResult to a HealthResponse.

    Args:
        result: Populated StatusResult from run_status_checks().

    Returns:
        HealthResponse with derived status field.
    """
    online_count = sum([result.lmstudio_online, result.ecb_online])
    if online_count == 2:
        status = "ok"
    elif online_count == 1:
        status = "degraded"
    else:
        status = "offline"

    return HealthResponse(
        status=status,
        version=result.quarq_version,
        lmstudio_online=result.lmstudio_online,
        lmstudio_model=result.lmstudio_active_model,
        fred_status=result.fred_status,
        ecb_online=result.ecb_online,
        corpus_chunks=result.corpus_chunks,
        corpus_docs=result.corpus_docs,
    )


@router.get("", response_model=HealthResponse)
async def get_health(request: Request) -> HealthResponse:
    """Return current health status from cached startup checks.

    Args:
        request: FastAPI Request (provides access to app.state).

    Returns:
        HealthResponse from app.state.status.
    """
    result: StatusResult = request.app.state.status
    return _status_to_response(result)


@router.get("/refresh", response_model=HealthResponse)
async def refresh_health(request: Request) -> HealthResponse:
    """Re-run all status checks and return fresh results.

    Args:
        request: FastAPI Request (provides access to app.state).

    Returns:
        Fresh HealthResponse after re-running all checks concurrently.
    """
    cfg = request.app.state.config
    result = await run_status_checks(cfg)
    request.app.state.status = result
    return _status_to_response(result)
