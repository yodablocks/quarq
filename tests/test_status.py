"""Tests for quarq.status module."""

from __future__ import annotations

import asyncio
from dataclasses import fields
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from quarq.config import DataConfig, LMStudioConfig, QuarqConfig, RAGConfig
from quarq.status import StatusResult, run_status_checks


def make_config(**overrides: object) -> QuarqConfig:
    """Build a QuarqConfig with optional section overrides."""
    cfg = QuarqConfig()
    for key, val in overrides.items():
        setattr(cfg, key, val)
    return cfg


def test_status_result_has_all_fields() -> None:
    """StatusResult dataclass must expose every required field."""
    required = {
        "lmstudio_online",
        "lmstudio_models",
        "lmstudio_active_model",
        "fred_status",
        "ecb_online",
        "corpus_docs",
        "corpus_chunks",
        "config_path",
        "quarq_version",
    }
    actual = {f.name for f in fields(StatusResult)}
    assert required <= actual, f"Missing fields: {required - actual}"


def test_all_offline_returns_clean_result() -> None:
    """When all HTTP calls fail, run_status_checks returns without raising."""
    cfg = make_config(
        lmstudio=LMStudioConfig(url="http://127.0.0.1:9999/v1"),
        data=DataConfig(fred_api_key="", fred_enabled=True, ecb_enabled=True, oecd_enabled=True),
        rag=RAGConfig(chroma_path="/nonexistent/path"),
    )

    import httpx

    async def raise_connect(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    transport = httpx.MockTransport(raise_connect)

    async def _run() -> StatusResult:
        async with httpx.AsyncClient(transport=transport, timeout=1.0) as client:
            return await run_status_checks(cfg, client=client)

    result = asyncio.get_event_loop().run_until_complete(_run())

    assert result.lmstudio_online is False
    assert result.lmstudio_models == []
    assert result.lmstudio_active_model == ""
    assert result.ecb_online is False
    assert result.corpus_docs == 0
    assert result.corpus_chunks == 0
    assert isinstance(result.config_path, Path)
    assert isinstance(result.quarq_version, str)


def test_fred_no_key_status() -> None:
    """When fred_api_key is empty, fred_status is 'no key'."""
    cfg = make_config(data=DataConfig(fred_api_key="", fred_enabled=True))

    async def _run() -> StatusResult:
        import httpx
        async with httpx.AsyncClient(timeout=1.0) as client:
            return await run_status_checks(cfg, client=client)

    with patch("quarq.status._check_lmstudio", new_callable=AsyncMock) as mock_lm, \
         patch("quarq.status._check_ecb", new_callable=AsyncMock) as mock_ecb:
        mock_lm.return_value = (False, [], "")
        mock_ecb.return_value = False

        result = asyncio.get_event_loop().run_until_complete(_run())

    assert result.fred_status == "no key"


def test_fred_connected_with_key() -> None:
    """When fred_api_key is set and the endpoint responds 200, fred_status is 'connected'."""
    cfg = make_config(data=DataConfig(fred_api_key="TESTKEY", fred_enabled=True))

    import httpx

    async def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"DATE,VALUE\n2024-01-01,4.5\n")

    transport = httpx.MockTransport(mock_handler)

    async def _run() -> StatusResult:
        async with httpx.AsyncClient(transport=transport, timeout=1.0) as client:
            return await run_status_checks(cfg, client=client)

    with patch("quarq.status._check_lmstudio", new_callable=AsyncMock) as mock_lm, \
         patch("quarq.status._check_ecb", new_callable=AsyncMock) as mock_ecb:
        mock_lm.return_value = (False, [], "")
        mock_ecb.return_value = False

        result = asyncio.get_event_loop().run_until_complete(_run())

    assert result.fred_status == "connected"
