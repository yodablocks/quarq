# llm/ — LLM backend layer

## Two agents, never interchangeable

| Agent | Config field | Purpose |
|---|---|---|
| Reporting | `config.llm.reporting_model` | Fast, structured narrative from metrics |
| Research | `config.llm.research_model` | Heavy, document-grounded RAG queries |

## Entry point

```python
from quarq.llm.lmstudio import get_llm
llm = get_llm(cfg, agent="research")
```

Always use `get_llm()`. Never instantiate `LMStudioLLM` or `ClaudeLLM` directly
outside of `get_llm()`.

## Fallback chain

1. LM Studio (local, `config.lmstudio.url`)
2. Claude API (cloud, `ANTHROPIC_API_KEY` env var)
3. `RAGError("No LLM backend available")` if neither works

## Model names

Model names come from config. Never hardcode a model name in any function call.
The only place model name strings appear is `config.py` default values.
