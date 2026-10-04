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

## Result (run 2026-10-04, results/run-20261004T033954.json)
Gold page inside the pool, out of 38 questions:
| Pool | Embedding | BM25 | Hybrid (RRF) | Hybrid gained | Hybrid lost |
|---|---|---|---|---|---|
| first 10 chunks | 36 | 36 | 35 | none | b2-013 |
| first 20 chunks | 36 | 37 | 37 | b1-020 | none |

End to end:
| Arm | Hit@1 | Hit@5 | MRR | Misses |
|---|---|---|---|---|
| embedding pool 10 + Jev (earlier run, baseline) | 36 | 36 | 0.947 | b1-020, b2-011 |
| hybrid pool 10 + Jev | 35 | 35 | 0.921 | b1-020, b2-011, b2-013 |
| hybrid pool 20 + Jev | 37 | 37 | 0.974 | b2-011 |
| hybrid pool 10 + bge (baseline bge: 33) | 32 | 35 | 0.869 | 6 |

Against the bands declared above: at the production pool size (N=10) hybrid is **Nothing** on the pool
(no gain, one loss) and **a regression** end to end (35 < 36). At N=20 it gains b1-020 and loses none
(37/38, MRR 0.974), for roughly twice the Jev calls (about $0.03 per run instead of $0.014).
b2-011 is in no pool, even at 20: neither embeddings nor BM25 find it.

Not a win: the gain is one question (b1-020), found only by widening the pool, on a gold set where one question is 2.6 points
and where I picked BM25 after looking at the misses. Why RRF pushed b2-013 out of the first 10 was not investigated.
Not run: embedding pool 20 + Jev (the pool numbers show it cannot beat 36, since the embedding pool holds the same 36 pages at N=10 and N=20),
BM25 or fusion tuning, any depth other than 40.
What this set cannot do: show a recall improvement. It has 2 failing questions, so there is almost nothing to fix and a lot of room to lose.

## Probe: one hard question at a time
`probe.py` checks a single question, so nothing runs the whole set unless asked. Light on the machine: thread count capped at 2,
query embedding and BM25 index cached in `.cache/` (a repeat probe loads no model), Jev is a cloud call, bge is never loaded.
```bash
python3 experiments/recall/probe.py "question" --gold file.pdf:12 [--gold ...] [--pool 10|20] [--no-jev]
python3 experiments/recall/probe.py --id h-001          # re-run a saved question
python3 experiments/recall/probe.py "question" --gold file.pdf:12 --save h-001   # append to hard_v1.jsonl
```
Saved questions use provenance `hand-written`, so `quarq eval --dataset` accepts the file. Gold pages use the PDF viewer index.
Checked on b1-020: embedding misses, BM25 ranks the gold page 4th, hybrid 14th, Jev ranks it first with a score of only 0.50
(the next best is 0.08). The first run took about 14 s wall and 4.7 s of CPU; the corpus is 2,571 chunks.
