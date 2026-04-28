"""FastAPI application factory with lifespan, CORS, and router registration."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from quarq import __version__
from quarq.api.routes.data import router as data_router
from quarq.api.routes.health import router as health_router
from quarq.api.routes.portfolio import router as portfolio_router
from quarq.api.routes.rag import router as rag_router
from quarq.config import load_config
from quarq.status import run_status_checks

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Run startup checks and store results in app.state.

    Args:
        app: The FastAPI application instance.

    Yields:
        None (control returns here on shutdown).
    """
    cfg = load_config()
    status = await run_status_checks(cfg)

    app.state.config = cfg
    app.state.status = status

    logger.info(
        "quarq API started — model=%s corpus=%d chunks/%d docs",
        status.lmstudio_active_model,
        status.corpus_chunks,
        status.corpus_docs,
    )
    yield
    logger.info("quarq API shutting down")


app = FastAPI(
    title="quarq API",
    description="French institutional portfolio analytics API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router, prefix="/health", tags=["health"])
app.include_router(portfolio_router, prefix="/portfolio", tags=["portfolio"])
app.include_router(rag_router, prefix="/rag", tags=["rag"])
app.include_router(data_router, prefix="/data", tags=["data"])


@app.get("/")
def root() -> dict:
    """Return API identity and docs link.

    Returns:
        Dict with name, version, and docs URL.
    """
    return {"name": "quarq", "version": __version__, "docs": "/docs"}
