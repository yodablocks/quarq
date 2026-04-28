# Style Rules

- Type hints on all parameters and return types
- Docstring on every public function: one-line summary, Args, Returns, Raises
- No magic numbers: constants go in config or `quarq/constants.py`
- PEP8, max line length 100, black-compatible
- Imports: stdlib, third-party, local (isort order)

## Error Handling

Custom classes: `ProviderError`, `RAGError`, `ConfigError`
Never bare `except`. User-facing errors via `rich` console, never raw traceback.

## Two-Agent Rule

- Reporting agent calls use `config.llm.reporting_model`
- Research agent calls use `config.llm.research_model`
- Never pass a model name as a string literal in a function call
