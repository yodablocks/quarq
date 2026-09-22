"""Smoke test for Open WebUI tool classes.

Run with:
    QUARQ_API_URL=http://127.0.0.1:8000 python demo/smoke_test_tools.py

Requires quarq serve to be running. The tools default to
host.docker.internal, which does not resolve from the host, so
QUARQ_API_URL must be set or every call fails as "server not running".

This hits the live stack. For contract checks that run without a server,
see tests/test_openwebui_tools.py.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

os.environ.setdefault("QUARQ_API_URL", "http://127.0.0.1:8000")

# Imported after QUARQ_API_URL is set, since the tools read it at construction.
from tools.quarq_portfolio_tool import Tools as PortfolioTools  # noqa: E402
from tools.quarq_rag_tool import Tools as RagTools  # noqa: E402


def _assert_reachable(result: str) -> None:
    """Fail loudly when the tool returned an error string instead of data.

    The tools catch connection errors and return a human-readable message, so
    a naive "output is non-empty" assertion passes even when nothing was
    reached. Check for those sentinels explicitly.
    """
    for sentinel in ("not running", "failed:", "Invalid input"):
        if sentinel in result:
            raise AssertionError(f"Tool did not reach the API: {result[:200]}")


def test_rag_tool() -> None:
    print("=== RAG Tool smoke test ===")
    tool = RagTools()
    result = tool.query_financial_documents(
        "What does the ECB Financial Stability Review say about concentration risk "
        "in European equity markets?"
    )
    print(result[:500])
    print("...")
    _assert_reachable(result)
    assert "Sources:" in result, "Expected source citations in a grounded answer"
    print("[PASS] RAG tool returned a cited answer\n")


def test_portfolio_tool() -> None:
    print("=== Portfolio Tool smoke test ===")
    tool = PortfolioTools()
    result = tool.analyse_portfolio(
        tickers="MC.PA,TTE.PA,AIR.PA,SAN.PA",
        weights="0.30,0.25,0.25,0.20",
        benchmark="^FCHI",
    )
    print(result)
    _assert_reachable(result)
    assert "CAGR" in result, "Expected CAGR in output"
    assert "Sharpe" in result, "Expected Sharpe in output"
    assert "Sharpe:        n/a" not in result, "Sharpe missing from API response"
    print("[PASS] Portfolio tool returned metrics\n")


if __name__ == "__main__":
    test_rag_tool()
    test_portfolio_tool()
    print("All smoke tests passed.")
