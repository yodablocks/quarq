"""quarq startup status checks.

All checks run concurrently via asyncio.gather. Total wall-clock time
is bounded by the shortest timeout that still catches real failures.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from quarq import __version__
from quarq.config import QuarqConfig, get_config_path
from quarq.constants import RAG_COLLECTION_NAME

logger = logging.getLogger(__name__)

_FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10"
_ECB_URL = "https://data-api.ecb.europa.eu/service/data/ILM/W.U2.EUR.AF.B01.A"
_TIMEOUT = 3.0


@dataclass
class StatusResult:
    """Aggregated result of all startup checks."""

    lmstudio_online: bool = False
    lmstudio_models: list[str] = field(default_factory=list)
    lmstudio_active_model: str = ""
    fred_status: str = "offline"  # "connected" | "no key" | "offline"
    ecb_online: bool = False
    corpus_docs: int = 0
    corpus_chunks: int = 0
    config_path: Path = field(default_factory=get_config_path)
    quarq_version: str = field(default_factory=lambda: __version__)


async def _check_lmstudio(cfg: QuarqConfig, client: httpx.AsyncClient) -> tuple[bool, list[str], str]:
    """Check LM Studio connectivity and available models.

    Args:
        cfg: Current QuarqConfig.
        client: Shared httpx async client.

    Returns:
        Tuple of (online, model_names, active_model).
    """
    url = f"{cfg.lmstudio.url}/models"
    try:
        resp = await client.get(url, timeout=_TIMEOUT)
        if resp.status_code == 200:
            data = resp.json()
            models: list[str] = [m.get("id", "") for m in data.get("data", [])]
            active = models[0] if models else ""
            return True, models, active
    except Exception:
        pass
    return False, [], ""


async def _check_fred(cfg: QuarqConfig, client: httpx.AsyncClient) -> str:
    """Check FRED API connectivity.

    Args:
        cfg: Current QuarqConfig.
        client: Shared httpx async client.

    Returns:
        "connected" | "no key" | "offline"
    """
    if not cfg.data.fred_api_key:
        return "no key"
    try:
        resp = await client.get(_FRED_URL, timeout=_TIMEOUT)
        if resp.status_code == 200:
            return "connected"
    except Exception:
        pass
    return "offline"


async def _check_ecb(cfg: QuarqConfig, client: httpx.AsyncClient) -> bool:
    """Check ECB SDW API connectivity.

    Args:
        cfg: Current QuarqConfig.
        client: Shared httpx async client.

    Returns:
        True if the endpoint responds 200.
    """
    try:
        resp = await client.get(
            _ECB_URL,
            headers={"Accept": "application/json"},
            timeout=_TIMEOUT,
        )
        return resp.status_code == 200
    except Exception:
        return False


def _check_corpus(cfg: QuarqConfig) -> tuple[int, int]:
    """Count documents and chunks in the local ChromaDB corpus.

    Args:
        cfg: Current QuarqConfig.

    Returns:
        Tuple of (doc_count, chunk_count). Both 0 if corpus is absent.
    """
    chroma_path = Path(cfg.rag.chroma_path).expanduser()
    if not chroma_path.exists():
        return 0, 0
    try:
        import chromadb

        chroma_client = chromadb.PersistentClient(path=str(chroma_path))
        try:
            col = chroma_client.get_collection(RAG_COLLECTION_NAME)
            chunk_count = col.count()
            if chunk_count > 0:
                results = col.get(include=["metadatas"])
                sources = {m.get("source", "") for m in (results.get("metadatas") or []) if m}
                return len(sources), chunk_count
        except Exception as exc:
            logger.warning("_check_corpus: collection query failed: %s", exc)
    except Exception as exc:
        logger.warning("_check_corpus: ChromaDB init failed: %s", exc)
    return 0, 0


async def run_status_checks(
    cfg: QuarqConfig,
    *,
    client: httpx.AsyncClient | None = None,
) -> StatusResult:
    """Run all startup checks concurrently and return aggregated results.

    Args:
        cfg: Current QuarqConfig.
        client: Optional shared httpx.AsyncClient (injected for testing).

    Returns:
        Populated StatusResult dataclass.
    """
    owns_client = client is None
    if owns_client:
        client = httpx.AsyncClient()

    try:
        lm_task = _check_lmstudio(cfg, client)
        fred_task = _check_fred(cfg, client)
        ecb_task = _check_ecb(cfg, client)

        (lm_online, lm_models, lm_active), fred_status, ecb_online = await asyncio.gather(
            lm_task, fred_task, ecb_task
        )
    finally:
        if owns_client:
            await client.aclose()

    corpus_docs, corpus_chunks = _check_corpus(cfg)

    return StatusResult(
        lmstudio_online=lm_online,
        lmstudio_models=lm_models,
        lmstudio_active_model=lm_active,
        fred_status=fred_status,
        ecb_online=ecb_online,
        corpus_docs=corpus_docs,
        corpus_chunks=corpus_chunks,
        config_path=get_config_path(),
        quarq_version=__version__,
    )
