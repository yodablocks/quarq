# Experiment 2: first-stage recall, BM25 plus embeddings

Follow-up to experiments/jev_rerank. There, both re-rankers missed the same 2 of 38 questions
(b1-020, b2-011) because the gold page was not in the embedding top 40. No re-ranker can fix that.
Question: does adding BM25 (keyword) candidates to the embedding candidates, fused by reciprocal
rank, put those pages in the pool without losing pages the embeddings already found?

Part of note 001 (from_Q2P repo). Throwaway code, outside the quarq package.

## Method
- BM25 written here (no new dependency): accent-folded lowercase `\w+` tokens, numbers kept,
  k1=1.5, b=0.75, over every chunk in the collection. Untuned.
- Fusion: reciprocal rank fusion, k=60, over the embedding top 40 chunks and the BM25 top 40.
  Untuned.
- The pool is the first N chunks handed to the re-ranker (N=10 as in production, and N=20).
- Everything after the pool is production code: re-rank, date-aware ordering, one chunk per page.

## Declared before the run (so the result can't be fitted)
I looked at the two missed questions (an exact-date factsheet weight question, and an ECB question
about tariff-related market instability) before choosing BM25. That is a bias toward them.
Hence the regression check across all 38 questions matters more than the two gains.

Bands, on gold-page-in-pool over all 38 questions, with the embedding pool as baseline:
- **Helps**: hybrid pool N=10 contains at least 1 more gold page than the embedding pool AND loses none.
- **Hurts**: hybrid loses 2 or more gold pages that the embedding pool had.
- **Nothing**: otherwise.
End to end, with the Jev v1 re-ranker (baseline 36/38 Hit@1): **no regression** means Hit@1 at least 36;
37 or 38 means improvement. With 38 questions and 2 possible gains, any gain is anecdotal, and I will say so.
Pool N=20 doubles Jev cost (about $0.03) and is run as a second knob, not a second chance:
both are reported.

## Run
`python3 experiments/recall/run.py`
