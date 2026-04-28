# api/ -- FastAPI REST wrapper

Thin wrapper over the pipeline. No business logic in routes.
Routes call pipeline functions, format results as Pydantic models.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | / | Root info |
| GET | /health | Status check (cached from startup) |
| GET | /health/refresh | Re-run checks live |
| POST | /portfolio/metrics | Compute risk metrics |
| GET | /portfolio/providers | List providers |
| POST | /rag/query | RAG question answering |
| POST | /rag/add | Index documents |
| GET | /rag/status | Corpus stats |
| DELETE | /rag/corpus | Clear corpus (requires X-Confirm-Delete: true) |
| GET | /data/{provider}/{series} | Fetch data series |
| GET | /data/providers | List providers |

## Config

All config comes from `app.state.config` (QuarqConfig).  
On provider failure: HTTP 503. On validation failure: HTTP 422. On missing resource: HTTP 404.

## Run

```bash
quarq serve
# or
uvicorn quarq.api.app:app --reload --port 8000
```

## Metric computation

`metrics.py` contains pure pandas/numpy implementations of Sharpe, drawdown, CAGR,
volatility, VaR, beta, and alpha. Never import external finance libraries here.

## LLM agents

Research agent (heavy model) handles RAG queries.  
Reporting agent (fast model) handles portfolio narrative generation.  
Never swap them. Config: `config.llm.research_model`, `config.llm.reporting_model`.
