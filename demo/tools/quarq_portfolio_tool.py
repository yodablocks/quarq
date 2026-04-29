"""
quarq Portfolio Tool for Open WebUI

Computes institutional risk metrics for a French equity portfolio and returns
a structured summary. Optionally generates an HTML report file.

quarq must be running: quarq serve (http://127.0.0.1:8000)
"""

import httpx
from datetime import date, timedelta


class Tools:
    def __init__(self):
        self.base_url = "http://127.0.0.1:8000"

    def analyse_portfolio(
        self,
        tickers: str,
        weights: str,
        start_date: str = "",
        end_date: str = "",
        benchmark: str = "^FCHI",
    ) -> str:
        """Compute risk metrics for a French equity portfolio.

        Use this tool when the user provides a list of French stock tickers
        and asks for portfolio analysis, risk metrics, Sharpe ratio, drawdown,
        CAGR, VaR, beta, or alpha.

        Args:
            tickers: Comma-separated list of tickers, e.g. "MC.PA,TTE.PA,AIR.PA"
            weights: Comma-separated weights summing to 1.0, e.g. "0.4,0.35,0.25"
            start_date: Start date YYYY-MM-DD. Defaults to 1 year ago.
            end_date: End date YYYY-MM-DD. Defaults to today.
            benchmark: Benchmark ticker. Default is ^FCHI (CAC 40).

        Returns:
            Structured portfolio metrics summary with institutional commentary.
        """
        try:
            ticker_list = [t.strip() for t in tickers.split(",")]
            weight_list = [float(w.strip()) for w in weights.split(",")]

            if not end_date:
                end_date = date.today().isoformat()
            if not start_date:
                start_date = (date.today() - timedelta(days=365)).isoformat()

            response = httpx.post(
                f"{self.base_url}/portfolio/metrics",
                json={
                    "tickers": ticker_list,
                    "weights": weight_list,
                    "start": start_date,
                    "end": end_date,
                    "benchmark": benchmark,
                    "include_narrative": False,
                },
                timeout=60.0,
            )
            response.raise_for_status()
            m = response.json()

            lines = [
                f"Portfolio: {', '.join(ticker_list)}",
                f"Period: {start_date} to {end_date}",
                f"Benchmark: {benchmark}",
                "",
                f"CAGR:          {m.get('cagr', 0)*100:.2f}%",
                f"Sharpe:        {m.get('sharpe', 0):.2f}",
                f"Max Drawdown:  {m.get('max_drawdown', 0)*100:.2f}%",
                f"VaR 95 (daily):{m.get('var_95', 0)*100:.2f}%",
                f"Volatility:    {m.get('volatility', 0)*100:.2f}%",
                f"Beta:          {m.get('beta', 0):.2f}",
                f"Alpha:         {m.get('alpha', 0)*100:.2f}%",
            ]

            narrative = m.get("narrative")
            if narrative:
                lines += ["", "Commentary:", narrative]

            lines += [
                "",
                "To generate a full HTML report with charts, run:",
                "quarq report --portfolio demo/portfolio.toml --open",
            ]

            return "\n".join(lines)

        except httpx.ConnectError:
            return (
                "quarq server is not running. "
                "Start it with: quarq serve"
            )
        except ValueError as exc:
            return f"Invalid input: {exc}"
        except Exception as exc:
            return f"Portfolio analysis failed: {exc}"
