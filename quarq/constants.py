"""Project-wide constants shared across quarq modules."""

RAG_COLLECTION_NAME = "quarq_rag_v1"

# Retrieval eval: k values reported by `quarq eval` (5 matches config.rag.top_k default).
DEFAULT_K_VALUES: tuple[int, ...] = (1, 3, 5)

# eval-gen skips chunks shorter than this (tables of contents, headers, page furniture).
EVAL_MIN_CHUNK_CHARS: int = 200

# Eval Markdown report: how many lowest-recall questions to list, and how many
# retrieved refs to show per question.
EVAL_REPORT_WORST_N: int = 5
EVAL_REPORT_REFS_SHOWN: int = 3

# quarq eval: how many unknown gold refs to list by name before summarizing the rest.
EVAL_UNKNOWN_REFS_SHOWN: int = 10
