"""Smoke test for Open WebUI tool classes. Run with: python demo/smoke_test_tools.py
Requires: quarq serve running on http://127.0.0.1:8000
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from tools.quarq_rag_tool import Tools as RagTools
from tools.quarq_portfolio_tool import Tools as PortfolioTools


def test_rag_tool() -> None:
    print("=== RAG Tool smoke test ===")
    tool = RagTools()
    result = tool.query_financial_documents(
        "What does the ECB Financial Stability Review say about concentration risk "
        "in European equity markets?"
    )
    print(result[:500])
    print("...")
    assert len(result) > 10, "Expected non-empty answer"
    print("[PASS] RAG tool returned output\n")


def test_portfolio_tool() -> None:
    print("=== Portfolio Tool smoke test ===")
    tool = PortfolioTools()
    result = tool.analyse_portfolio(
        tickers="MC.PA,TTE.PA,AIR.PA,SAN.PA",
        weights="0.30,0.25,0.25,0.20",
        benchmark="^FCHI",
    )
    print(result)
    assert "CAGR" in result, "Expected CAGR in output"
    assert "Sharpe" in result, "Expected Sharpe in output"
    print("[PASS] Portfolio tool returned metrics\n")


if __name__ == "__main__":
    test_rag_tool()
    test_portfolio_tool()
    print("All smoke tests passed.")
