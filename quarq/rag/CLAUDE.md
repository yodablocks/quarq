# rag/ — retrieval-augmented generation layer

Chunk size 512 tokens. Overlap 64 tokens.
Collection: `quarq_rag_v1`

## Files

| File | Purpose |
|---|---|
| `loader.py` | PDF extraction, chunking, metadata attachment |
| `embedder.py` | multilingual-e5-large, passage/query prefixes required |
| `store.py` | ChromaDB wrapper, upsert and query |
| `retriever.py` | top-k with min_similarity filter, metadata filters |
| `generator.py` | prompt builder, dual-agent LLM routing, citations |

## Embedder prefix rule (enforce strictly)

```
Passages: "passage: " + text
Queries:  "query: "   + text
```

Wrong prefixes silently degrade retrieval quality. This is non-negotiable.

## Required metadata per chunk

All five fields must be present on every Document before upsert:

| Field | Type | Values |
|---|---|---|
| `source` | str | filename only |
| `doc_type` | str | `ecb_fsr` \| `amf_sfdr` \| `prospectus` \| `factsheet` \| `bdf_fsr` \| `macro` |
| `date` | str | publication date or "unknown" |
| `page` | int | page number |
| `chunk_id` | str | sha256 of content |

## Retrieval defaults (from config)

- `top_k`: 5
- `min_similarity`: 0.35

## Generation rule

Never hallucinate citations. If no relevant chunk is found, the answer must
say so explicitly. The system prompt enforces this — do not modify it without
updating this doc.
