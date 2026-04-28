# quarq

French institutional portfolio analytics: live data pipeline, RAG over
financial documents, dual-agent LLM narrative layer.

## Stack

- Python 3.11+, type hints required everywhere
- Data: yfinance, FRED REST, ECB SDW REST, OECD REST (direct, no framework)
- RAG: sentence-transformers (multilingual-e5-large), ChromaDB (local)
- LLM reporting agent: fast model via LM Studio (qwen/qwen3.5-9b default)
- LLM research agent: heavy model via LM Studio (qwen/qwen3.6-27b default)
- LLM cloud fallback: Claude API (claude-sonnet-4-20250514)
- API: FastAPI, uvicorn, Pydantic v2
- Report: Plotly 6 (use go.Heatmap and go.Treemap, px wrappers broken in v6)
- CLI: rich (Phase 2), textual (Phase 3+)

## Forbidden

- No OpenBB: AGPLv3 licence risk + SDK instability
- No LangChain: opaque internals, breaking changes
- No LlamaIndex: same
- No hardcoded model names outside config.py and llm/ backends
- No hardcoded financial constants outside config

## Run

```bash
pip install -e .
quarq
quarq status
pytest tests/
```

Config: `~/.quarq/config.toml`
Local overrides: `CLAUDE.local.md` (gitignored)

## Two LLM Agents

- **Reporting agent:** structured metrics to narrative (fast, runs every report)
- **Research agent:** question to RAG retrieval to answer (heavy, on demand)
- Model names come from config. Never hardcode.

## Architecture

```
ingest/ -> pipeline/ -> rag/ -> llm/ -> report/
```

Each layer has its own CLAUDE.md with interface contracts.

@.claude/rules/pipeline-rules.md
@.claude/rules/rag-rules.md
@.claude/rules/style-rules.md
