# quarq

**French institutional portfolio analytics with RAG and local LLM.**

> `pip install quarq`

---

## What it is

`quarq` is a CLI tool for institutional portfolio analysis. It combines a live data pipeline (French equities, ECB macro, FRED rates) with a RAG engine over financial documents (SFDR filings, ECB FSR, fund prospectuses) and a local LLM for plain-English narrative generation.

Your portfolio data never leaves your machine.

## Status

Work in progress. v0.1.0 coming soon.

### What works now

```python
from datetime import date
from quarq.ingest.equity import EquityProvider
from quarq.ingest.fred import get_risk_free_rate
from quarq.ingest.ecb import get_ecb_rate
from quarq.ingest import get_provider
from quarq.config import load_config

# Live equity prices (.PA tickers, ^FCHI benchmark)
p = EquityProvider()
df = p.fetch("MC.PA", date(2025, 1, 1), date(2025, 12, 31))

# Risk-free rate (FRED OAT10Y, falls back to config if no key)
cfg = load_config()
rfr = get_risk_free_rate(cfg)  # e.g. 0.032

# ECB policy rate (no key required)
ecb = get_ecb_rate(cfg)  # e.g. 0.045

# Provider registry
equity = get_provider("equity")   # EquityProvider
fred   = get_provider("fred")     # FREDProvider
ecb_p  = get_provider("ecb")      # ECBProvider
oecd   = get_provider("oecd")     # OECDProvider
```

Set `FRED_API_KEY` in `~/.quarq/config.toml` for live macro rates. All other providers work without a key.

## Stack

- Data: yfinance, FRED, ECB SDW, OECD (direct REST, no framework dependency)
- RAG: sentence-transformers + ChromaDB (local, persistent)
- LLM: Qwen3 / Mistral via LM Studio (local inference)
- Report: Plotly 6 + Jinja2, PDF via playwright
- CLI: textual + rich

## Usage (coming in v0.1.0)

quarq status
quarq report --portfolio ./portfolio.toml --format pdf
quarq query "What does the ECB FSR say about CAC 40 concentration?"
quarq rag add ./docs/

## Author

zkmarc — https://github.com/yodablocks
