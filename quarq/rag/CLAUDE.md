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
| `manifest.py` | corpus manifest: per-document period, publication date, doc_type overrides |

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
| `date` | str | publication date (from the manifest when set; otherwise PDF creation date, a filename year, or "unknown") |
| `page` | int | page number |
| `chunk_id` | str | sha256 of content |

Optional, set by the corpus manifest (`quarq_manifest.toml` next to the PDFs):

| Field | Type | Values |
|---|---|---|
| `period_start` | str | ISO date, first day of the period the document covers |
| `period_end` | str | ISO date, last day of that period |

`quarq rag manifest <folder>` writes these onto chunks already indexed, without
re-embedding. `quarq rag add` applies them when indexing.

## Retrieval defaults (from config)

- `top_k`: 5
- `min_similarity`: 0.35

## Generation rule

Never hallucinate citations. If no relevant chunk is found, the answer must
say so explicitly. The system prompt enforces this — do not modify it without
updating this doc.
