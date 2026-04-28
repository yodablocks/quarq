# quarq/ingest — Data Provider Layer

Fetches time-series data from external sources and normalises it into a
single DataFrame schema consumed by the pipeline.

## BaseProvider Interface

Every provider in this directory subclasses `BaseProvider` (base.py):

```python
def fetch(self, series_id: str, start: date, end: date) -> pd.DataFrame
```

Returned DataFrame contract:
- Index: `DatetimeIndex`
- Columns: `value` (float64), `series_id` (str), `source` (str)
- Raises `ProviderError` on any failure. Never returns partial data silently.

## Cache

`cache.py` stores pickled DataFrames in `~/.quarq/cache/`.
TTL values:
- equity: 3600 s (1 hour)
- macro (fred, ecb, oecd): 86400 s (24 hours)

## Provider Registry

`PROVIDER_REGISTRY` in `__init__.py` maps provider names to classes.
`get_provider(name)` instantiates by name; raises `ProviderError` for unknowns.

## Known Failure Modes

| Provider | Failure mode | Behaviour |
|----------|-------------|-----------|
| equity   | Bad ticker / empty yfinance response | Raises `ProviderError` |
| equity   | < 5 rows returned | Logs warning, continues |
| fred     | No API key | Raises `ProviderError` immediately |
| fred     | HTTP error | Raises `ProviderError` |
| ecb      | Endpoint unreachable | Raises `ProviderError` |
| ecb      | Malformed JSON | Raises `ProviderError` |
| oecd     | Unknown series_id | Raises `ProviderError` before any HTTP call |
| oecd     | Endpoint unreachable | Raises `ProviderError` |

## Adding a Provider

1. Create `quarq/ingest/<name>.py` subclassing `BaseProvider`
2. Add to `PROVIDER_REGISTRY` in `__init__.py`
3. Add mocked HTTP tests in `tests/test_ingest.py`
4. Document cache TTL and known failures in this file
