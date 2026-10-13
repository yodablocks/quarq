# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-10-13

First release on PyPI with the real package: 0.0.1 to 0.0.4 were placeholders that claimed
the name, and 0.1.0 was tagged but never uploaded. This release includes everything in 0.1.0
below plus the changes listed here.

### Added

**Retrieval evaluation**
- `quarq eval` scores the real retriever against a human-reviewed gold set: precision, recall,
  hit rate and MRR, with rich, JSON and Markdown reports.
- Gold set of 38 reviewed questions, including 10 that depend on the year asked about. Each row
  carries its provenance, and refs missing from the corpus are rejected.
- `quarq eval-gen` drafts new gold questions for a person to review; it never overwrites.
- Every eval run compares the vector index with exact search, so index recall is measured.

**RAG**
- Cross-encoder re-ranking of the top candidates.
- Questions that name an exact date favour documents covering that date.
- Answers flag figures the model was not shown.
- A corpus manifest records each document's period and dates (`quarq rag manifest`).
- `quarq rag coverage` and `quarq rag dedupe`; PDF pages with no text layer are reported.

### Changed
- The collection is now `quarq_rag_v2`, created with explicit HNSW settings so retrieval is
  exact. `quarq rag migrate` copies an existing `quarq_rag_v1` collection without re-embedding.
- Search returns one chunk per page, so the top k results are distinct pages.

### Fixed
- Duplicate chunks are removed, and re-indexing no longer adds copies.
- Secrets in the config are stored as `SecretStr` so they never print, and saving with an
  environment override keeps the stored secret.
- Eval input errors name the invalid field and fail with a clear message.

### Packaging
- Published to PyPI from a tagged GitHub Actions release, using trusted publishing.
- The version is defined once, in `quarq/__init__.py`.

## [0.1.0] - 2026-09-22

First tagged release. Everything below landed across phases 2 through 7.

### Added

**Data pipeline**
- `BaseProvider` interface with a provider registry (`get_provider`).
- Equity prices via yfinance, with `.PA` tickers and the `^FCHI` benchmark.
- FRED (OAT10Y risk-free rate), ECB SDW (policy rate), and OECD providers over
  direct REST. Only FRED needs an API key; the others work without one.
- On-disk response caching keyed by `provider:series_id:start:end`.

**RAG**
- ChromaDB vector store (`quarq_rag_v1`) with persistent local storage.
- `intfloat/multilingual-e5-large` embeddings, chosen for French regulatory text.
- PDF ingestion via pdfplumber, 512-token chunks with 64-token overlap.
- Retrieval with a top-k and minimum-similarity floor; every answer carries
  source and page citations.

**Portfolio analytics**
- CAGR, Sharpe, max drawdown, volatility, VaR 95, beta, and alpha.
- `PortfolioSpec` and a TOML portfolio loader with validation.

**LLM layer**
- Two-agent split: a fast reporting agent for narrative, a heavier research
  agent for RAG answers. Model names come from config, never hardcoded.
- LM Studio backend with automatic discovery of the active model.
- Claude API fallback via `ANTHROPIC_API_KEY`.

**Reporting**
- HTML reports with Plotly 6 charts (`go.Heatmap`, `go.Treemap`) and Jinja2.
- Optional LLM-generated narrative.

**API**
- FastAPI server (`quarq serve`) exposing `/health`, `/portfolio/metrics`,
  `/rag/query`, `/rag/add`, `/rag/status`, `/data/{provider}/{series_id}`,
  and `/report`.

**CLI**
- `quarq status`, `version`, `query`, `rag add|status`, `config`, `serve`,
  and `report`.

**Open WebUI integration**
- Two uploadable tool classes exposing RAG queries and portfolio analysis to
  a local chat UI, plus a setup guide and demo launcher.

### Fixed

- Sharpe ratio always displayed as `0.00` in Open WebUI: the portfolio tool
  read `sharpe` from the response, but the API returns `sharpe_ratio`.
- The RAG tool sent `n_results`, but the API expects `k`. Pydantic dropped the
  unknown field, so the requested page size silently never applied.
- Null `beta` and `alpha` (returned whenever benchmark data is unavailable)
  raised `TypeError` during formatting and surfaced as
  "Portfolio analysis failed". They now render as `n/a`.
- The demo smoke test passed while reaching nothing: the tools return an error
  string on connection failure, and the only assertion was that the output was
  non-empty.
- An empty RAG corpus (HTTP 422) surfaced in Open WebUI as an opaque
  "RAG query failed" string instead of pointing at `quarq rag add`.
- Jinja2 autoescaping broke Plotly script tags in rendered reports.
- `quarq report --open` produced a relative `file://` URL.

### Security

- `FRED_API_KEY` can now be supplied via the environment instead of being
  stored in `~/.quarq/config.toml`. `save_config` strips any value that came
  from the environment before writing, so a load-mutate-save round trip (as
  performed by `quarq config --set-lmstudio-url`) cannot persist it.
- Generated HTML reports are gitignored. They embed portfolio holdings and
  were previously untracked rather than ignored.

### Known limitations

- The Open WebUI, Docker, and LM Studio integration path is covered by mocked
  contract tests only. End-to-end behaviour against a live stack is verified
  by hand via `demo/smoke_test_tools.py`.
- PDF report output requires the `full` extra (`pip install 'quarq[full]'`)
  for playwright.
- Not published to PyPI; install from source (fixed in 0.2.0).

[0.2.0]: https://github.com/yodablocks/quarq/releases/tag/v0.2.0
[0.1.0]: https://github.com/yodablocks/quarq/releases/tag/v0.1.0
