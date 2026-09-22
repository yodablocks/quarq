# quarq

**French institutional portfolio analytics with RAG and local LLM.**

Portfolio risk metrics, retrieval-grounded answers over institutional
documents, and plain-English narrative, all running on your own machine.
Your portfolio data never leaves it.

---

## Status

**v0.1.0** — first tagged release. Install from source; not yet on PyPI.

```bash
git clone https://github.com/yodablocks/quarq
cd quarq
pip install -e .
```

Python 3.11+. For PDF report output, install the extras: `pip install -e '.[full]'`.

## Quick start

```bash
quarq status                                  # check providers, LM Studio, corpus
quarq rag add ./docs/                         # index PDFs into the corpus
quarq query "What does the ECB FSR say about CAC 40 concentration?"
quarq report --portfolio ./demo/portfolio.toml --open
quarq serve                                   # REST API on 127.0.0.1:8000
```

### Commands

| Command | What it does |
|---|---|
| `quarq status` | Provider, LM Studio, and corpus health table |
| `quarq version` | Print version |
| `quarq query <question>` | Ask the RAG corpus. `--doc-type`, `--k` |
| `quarq rag add <path>` | Index a PDF or folder |
| `quarq rag status` | Corpus statistics |
| `quarq config --set-lmstudio-url <url>` | Point at your LM Studio instance |
| `quarq serve` | FastAPI server. `--host`, `--port`, `--reload` |
| `quarq report --portfolio <toml>` | Generate a report. `--format html\|pdf\|json`, `--narrative`, `--open` |

## Configuration

Config lives at `~/.quarq/config.toml`, created with defaults on first run.

Secrets are better supplied through the environment than stored on disk:

```bash
export FRED_API_KEY=...        # live macro rates; overrides the config value
export ANTHROPIC_API_KEY=...   # cloud fallback when LM Studio is unavailable
```

`FRED_API_KEY` is read at load time and never written back to
`config.toml`. Without it, quarq falls back to the configured risk-free rate.
ECB and OECD need no key.

## Library use

```python
from datetime import date
from quarq.ingest.equity import EquityProvider
from quarq.ingest.fred import get_risk_free_rate
from quarq.config import load_config

p = EquityProvider()
df = p.fetch("MC.PA", date(2025, 1, 1), date(2025, 12, 31))

cfg = load_config()
rfr = get_risk_free_rate(cfg)   # e.g. 0.032
```

## Metrics

CAGR, Sharpe ratio, maximum drawdown, volatility, VaR 95, beta, and alpha,
computed against a configurable benchmark (default `^FCHI`, the CAC 40).

## Open WebUI

quarq ships two tool classes that expose RAG queries and portfolio analysis
inside a local Open WebUI chat. See
[`demo/OPEN_WEBUI_SETUP.md`](demo/OPEN_WEBUI_SETUP.md).

The tools call quarq over HTTP and default to `host.docker.internal:8000`,
which is what resolves from inside the Open WebUI container. Running them
from the host instead, set `QUARQ_API_URL=http://127.0.0.1:8000`.

> **Note:** the tool files are *uploaded* into Open WebUI, not imported from
> this repo. After pulling changes to `demo/tools/*.py`, re-upload both files
> in Admin → Tools or your instance keeps running the old copies.

## Stack

- Data: yfinance, FRED, ECB SDW, OECD (direct REST, no framework dependency)
- RAG: sentence-transformers + ChromaDB, local and persistent
- LLM: Qwen3 via LM Studio, with a Claude API fallback
- Report: Plotly 6 + Jinja2, PDF via playwright
- API: FastAPI + Pydantic v2
- CLI: rich

## Tests

```bash
pytest tests/          # 93 tests, no network or server required
```

The Open WebUI tools are covered by mocked contract tests that validate
request bodies against the real Pydantic models. End-to-end verification
against a live stack is manual:

```bash
quarq serve &
QUARQ_API_URL=http://127.0.0.1:8000 python demo/smoke_test_tools.py
```

## Licence

MIT.

## Author

zkmarc — https://github.com/yodablocks
