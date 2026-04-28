# Pipeline Rules

Every data provider in `ingest/` implements `BaseProvider` (`base.py`):

```python
class BaseProvider(ABC):
    @abstractmethod
    def fetch(self, series_id: str, start: date, end: date) -> pd.DataFrame: ...
```

Returned DataFrame: `DatetimeIndex`, columns `value` (float64), `series_id` (str),
`source` (str). Raise `ProviderError` on any failure. Never return partial data.
Cache key format: `"{provider_name}:{series_id}:{start}:{end}"`

## Adding a Provider

1. Create `quarq/ingest/<name>.py` inheriting `BaseProvider`
2. Register in `PROVIDER_REGISTRY` in `quarq/ingest/__init__.py`
3. Add test in `tests/test_ingest.py` with mocked HTTP
4. Document in config schema
