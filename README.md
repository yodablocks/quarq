# quarq

**French institutional portfolio analytics with RAG and local LLM.**

> `pip install quarq`

---

## What it is

`quarq` is a CLI tool for institutional portfolio analysis. It combines a live data pipeline (French equities, ECB macro, FRED rates) with a RAG engine over financial documents (SFDR filings, ECB FSR, fund prospectuses) and a local LLM for plain-English narrative generation.

Your portfolio data never leaves your machine.

## Status

Work in progress. v0.1.0 coming soon.

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
