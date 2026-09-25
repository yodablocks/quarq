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

# Retriever: fetch this many times top_k candidate chunks before keeping the best chunk
# per (source, page), so k distinct pages can still be returned when one page has several
# matching chunks. The 2026-09-25 baseline saw up to 4 chunks from a single page in a top 5.
RETRIEVAL_OVERFETCH_FACTOR: int = 4

# Corpus manifest: per-document metadata (period covered, publication date, doc_type)
# kept next to the PDFs. `quarq rag add` reads it from the folder being indexed (or the
# file's parent folder); `quarq rag manifest` applies it to chunks already indexed.
CORPUS_MANIFEST_FILENAME: str = "quarq_manifest.toml"

# Allowed doc_type values. The loader's filename rules and manifest overrides must use
# one of these, so a typo can't silently create a new type that breaks doc_type filters.
DOC_TYPES: frozenset[str] = frozenset(
    {"ecb_fsr", "bdf_fsr", "amf_sfdr", "prospectus", "factsheet", "macro"}
)

# Retriever: always ask the store for at least this many candidates. ChromaDB's HNSW search
# only explores about as many neighbours as results requested (its ef_search setting had no
# effect in testing), so asking for 20 missed true top-20 chunks on 7 of 28 gold questions;
# asking for 500 missed none, at about 4 ms per query on 4,235 chunks. Re-check with the
# exact-search comparison if the corpus grows by an order of magnitude.
RETRIEVAL_CANDIDATE_POOL: int = 500
