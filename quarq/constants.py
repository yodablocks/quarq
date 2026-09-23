"""Project-wide constants shared across quarq modules."""

RAG_COLLECTION_NAME = "quarq_rag_v1"

# Retrieval eval: k values reported by `quarq eval` (5 matches config.rag.top_k default).
DEFAULT_K_VALUES: tuple[int, ...] = (1, 3, 5)

# eval-gen skips chunks shorter than this (tables of contents, headers, page furniture).
EVAL_MIN_CHUNK_CHARS: int = 200
