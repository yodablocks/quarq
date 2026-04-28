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
- euronext: 3600 s (1 hour — index composition can change intraday)
- macro (fred, ecb, oecd, eurostat, bdf, amf, cdp): 86400 s (24 hours)

## Provider Registry

`PROVIDER_REGISTRY` in `__init__.py` maps provider names to classes.
`get_provider(name)` instantiates by name; raises `ProviderError` for unknowns.

| Key | Class | Module | Series |
|-----|-------|--------|--------|
| equity | EquityProvider | equity.py | Any yfinance ticker |
| fred | FREDProvider | fred.py | Any FRED series code |
| ecb | ECBProvider | ecb.py | MRR_FR |
| oecd | OECDProvider | oecd.py | CPI_FR, GDP_FR |
| eurostat | EurostatProvider | eurostat.py | HICP_FR, UNEMP_FR, GDP_FR_Q |
| bdf | BDFProvider | bdf.py | OAT_10Y_FR, CREDIT_FR |
| euronext | EuronextProvider | euronext.py | CAC40_CONSTITUENTS, CAC40_PRICE |
| amf | AMFProvider | amf.py | SFDR_FUNDS |
| cdp | CDPProvider | cdp.py | CDP_SCORES, or any company name |

## Euronext API Note

Euronext API endpoint paths change occasionally. If `EuronextProvider.fetch` raises
with "endpoint not found", check `https://api.euronext.com` for the current paths and
update `_EURONEXT_BASE` in `quarq/ingest/euronext.py`.

## AMF File Note

The AMF SFDR Excel file URL and column structure change periodically.
If `AMFProvider.fetch` raises, check `https://www.amf-france.org` for the current
download URL and update `_AMF_SFDR_URL` in `quarq/ingest/amf.py`.

## CDP Socrata Note

The CDP Socrata endpoint (`data.cdp.net/resource/shjr-9ej4.csv`) is rate-limited
and may be unavailable. On failure, `CDPProvider.fetch` raises `ProviderError`
with instructions to download manually from `https://data.cdp.net`.

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
| eurostat | Unknown series_id | Raises `ProviderError` before any HTTP call |
| eurostat | HTTP error | Raises `ProviderError` |
| eurostat | Malformed JSON | Raises `ProviderError` |
| bdf      | Unknown series_id | Raises `ProviderError` before any HTTP call |
| bdf      | HTTP 404 | Raises `ProviderError` with dataset ID + webstat portal URL |
| bdf      | Other HTTP error | Raises `ProviderError` mentioning endpoint |
| bdf      | Empty/malformed JSON | Raises `ProviderError` |
| euronext | Unknown series_id | Raises `ProviderError` before any HTTP call |
| euronext | HTTP 404 | Raises `ProviderError` with endpoint + docs URL |
| euronext | Other HTTP error | Raises `ProviderError` mentioning endpoint |
| euronext | Malformed JSON / no price | Raises `ProviderError` |
| amf      | Unknown series_id | Raises `ProviderError` before any HTTP call |
| amf      | Download failure | Raises `ProviderError` with URL + amf-france.org suggestion |
| amf      | Excel parse failure | Raises `ProviderError` with file URL |
| amf      | < 10 rows in file | Logs warning, continues |
| cdp      | Unknown series type | Raises `ProviderError` |
| cdp      | Connection error | Raises `ProviderError` with data.cdp.net manual download URL |
| cdp      | No data for company | Raises `ProviderError` with company name |
| cdp      | CSV schema changed | Raises `ProviderError` listing found columns |

## Adding a Provider

1. Create `quarq/ingest/<name>.py` subclassing `BaseProvider`
2. Add to `PROVIDER_REGISTRY` in `__init__.py`
3. Add mocked HTTP tests in `tests/test_ingest.py`
4. Document cache TTL and known failures in this file

## BdF API Note

The Banque de France webstat API endpoint paths change occasionally.
If `BDFProvider.fetch` raises with "endpoint not found", check
https://webstat.banque-france.fr for the current dataset ID and update
`_SERIES_MAP` in `quarq/ingest/bdf.py`.
