"""
quarq Portfolio Tool for Open WebUI

Computes institutional risk metrics for a French equity portfolio and returns
a structured summary. Optionally generates an HTML report file.

quarq must be running: quarq serve (http://127.0.0.1:8000)
"""

import os
from datetime import date, timedelta

import httpx


class Tools:
    def __init__(self):
        # host.docker.internal resolves from inside the Open WebUI container.
        # Override with QUARQ_API_URL when running on the host (e.g. smoke tests).
        self.base_url = os.environ.get(
            "QUARQ_API_URL", "http://host.docker.internal:8000"
        )

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

            def _pct(key: str) -> str:
                """Format a ratio as a percentage, or n/a when unavailable."""
                value = m.get(key)
                return f"{value * 100:.2f}%" if value is not None else "n/a"

            def _num(key: str) -> str:
                """Format a plain float, or n/a when unavailable."""
                value = m.get(key)
                return f"{value:.2f}" if value is not None else "n/a"

            lines = [
                f"Portfolio: {', '.join(ticker_list)}",
                f"Period: {start_date} to {end_date}",
                f"Benchmark: {benchmark}",
                "",
                f"CAGR:          {_pct('cagr')}",
                f"Sharpe:        {_num('sharpe_ratio')}",
                f"Max Drawdown:  {_pct('max_drawdown')}",
                f"VaR 95 (daily):{_pct('var_95')}",
                f"Volatility:    {_pct('volatility')}",
                f"Beta:          {_num('beta')}",
                f"Alpha:         {_pct('alpha')}",
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
        # Broad catch is deliberate: Open WebUI renders whatever string the tool
        # returns, so an escaping exception would surface as an opaque UI error.
        except Exception as exc:  # noqa: BLE001
            return f"Portfolio analysis failed: {exc}"
