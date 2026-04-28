# RAG Rules

Chunk size 512 tokens. Overlap 64 tokens.

## Required Metadata per Chunk

| Field | Type | Values |
|---|---|---|
| `source` | str | filename |
| `doc_type` | str | `ecb_fsr` \| `amf_sfdr` \| `prospectus` \| `factsheet` \| `macro` |
| `date` | str | document publication date |
| `page` | int | page number |
| `chunk_id` | str | sha256 of content |

Never embed without all five fields.

## Retrieval

- top-k: 5
- minimum similarity: 0.35
- Every answer returns source citations: `list[tuple[source, page, snippet]]`

## ChromaDB

Collection name: `quarq_rag_v1`
Bump suffix if schema changes require full re-index.

## Embedding Model

`intfloat/multilingual-e5-large` — multilingual, handles French regulatory text.
Model name comes from `config [embedder] model`. Never hardcode.
