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
b2-011 is in no HYBRID pool, even at 20. CORRECTION (same day, found with probe.py): I first wrote that neither embeddings nor BM25 find it. That was wrong for BM25: BM25 alone ranks the gold page first, and my RRF fusion pushed it to 20th because the embedding list does not contain it.

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

### Correction and a lead (found with probe.py on b2-011)
BM25 alone, gold page in the first 10 / 20 chunks: misses b2-013 and b2-017 at 10, only b2-013 at 20 (so 37/38 at 20).
Hybrid (RRF) at 20: 37/38, missing b2-011. Embedding: 36/38 at both depths. The methods fail on different questions,
and the fusion step is what loses b2-011. A union (top 10 of each list) or a keyword-first merge might reach 38/38 at a pool of 20.
NOT tried. It would be tuning the merge on the same 38 questions after seeing which ones fail, so a gain there would not be evidence.

## Hard set v1 probe results (11 questions: h-001, h-002 known misses; h-003..h-011 drafted 2026-10-04)
Gold page rank = rank among distinct pages in each candidate list (depth 40). Pool = first N chunks of the RRF hybrid, handed to Jev.
| ID | Source language | Embedding | BM25 | Hybrid | Gold in pool 10 | Gold in pool 20 | Jev rank of gold (score) |
|---|---|---|---|---|---|---|---|
| h-001 | EN table | miss | 4 | 14 | no | yes | 1 (0.50) |
| h-002 | EN | miss | 1 | 20 | no | no | not in pool |
| h-003 | EN | 2 | 2 | 2 | yes | yes | 1 (0.87) |
| h-004 | EN | 2 | 6 | 2 | yes | yes | 1 (0.96) |
| h-005 | EN table | 1 | 1 | 1 | yes | yes | 1 (0.98) |
| h-006 | EN | 3 | 1 | 1 | yes | yes | 1 (0.94) |
| h-007 | EN | 1 | 1 | 1 | yes | yes | 1 (0.95) |
| h-008 | FR table | miss | 6 | 11 | no | yes | 1 (0.97) |
| h-009 | FR table | 14 | miss | 28 | no | no | not in pool |
| h-010 | FR | 4 | miss | 12 | no | yes | 1 (0.86) |
| h-011 | FR | miss | miss | miss (>72) | no | no | not in pool |

Gold in pool: 5/11 at 10, 8/11 at 20. Whenever the gold page is in the pool, Jev ranks it first (8 of 8). Jev cost for the 9 new questions: about $0.004.
Observations (n is small, questions drafted by me, nothing tuned):
- All 4 questions whose answer is on a French page, asked in English (h-008..h-011), do worse than the 5 English-page ones: embedding ranks 4 / 14 / miss / miss, and BM25 cannot cross languages (h-009, h-010, h-011 miss). Hypothesis, not tested: a cross-language effect. Untested: the same questions in French.
- RRF buried a page that a single method had in its top 6 in 4 of 11 questions (h-001, h-002, h-008, h-010). Fusion looks like the weak link again, on questions I had not seen before for h-008 and h-010.
- h-009 and h-011 are in no pool at all.
Not done: union merge, French rewrites, a human check of the questions (rows are still `synthetic-draft`).

## Step 1 after the hard set: union merge (round-robin of embedding and BM25 chunks)
Prediction written before the run, judged on "gold page in the first 10 chunks": union catches h-001, h-002, h-010 and probably h-008;
it does not catch h-009 or h-011. Anything else means the prediction was wrong.
Run: `python3 experiments/recall/probe.py --id h-001 --pool 10 --merge union --no-jev`

### Result of step 1 (union merge, hard set v1, 11 questions, no Jev calls)
Gold page in the first 10 chunks: **union 9/11** (misses h-009, h-011), versus RRF 5/11 and the best single method alone varying by question.
Same at pool 20 (9/11). The prediction held exactly: caught h-001, h-002, h-008, h-010; missed h-009, h-011.
Weak evidence, three reasons: I made the prediction after seeing the per-method ranks in the table above, so it is a consistency check, not a blind test;
h-001 and h-002 were the very questions that motivated the idea; the set is 11 questions I drafted.
NOT yet checked: regression over the original 38 questions (the earlier band said a merge must not lose pages the embeddings already found).
Until that is run, nothing here says union is better than the embedding pool in general.

### Step 1b: 38-question regression check, condition written before the run
Pass = union loses no gold page that the embedding pool already had in the first 10 chunks (N=10; N=20 reported too).
Any loss is named by id and counts as a regression. Passing means "did not break anything", not "is better".
Run: `python3 experiments/recall/regress.py` (no Jev calls, no bge, one embedder load).

### Result of step 1b (results/regress-38.json)
Gold page in the first N chunks, over the 38 original questions:
| Pool | Embedding | BM25 | RRF | Union | Union lost | Union gained |
|---|---|---|---|---|---|---|
| N=10 | 36 | 36 | 35 | **38** | none | b1-020, b2-011 |
| N=20 | 36 | 37 | 37 | **38** | none | b1-020, b2-011 |
Condition met: the union loses nothing the embedding pool had. The RRF rows reproduce the earlier run (35 at N=10 losing b2-013, 37 at N=20), a sanity check on the harness.
Caveats: the two gains are the very questions that motivated the merge, so they are in-sample; the out-of-sample evidence is h-008 and h-010 on the hard set.
Pool membership is not Hit@1. At N=10 the union gives the re-ranker about 5 embedding chunks plus 5 BM25 chunks instead of 10 embedding chunks, which could change the final order. Not measured yet.

### Step 1c: union + Jev end to end on the 38 questions, bands written before the run
Baseline: Jev v1 over the embedding pool of 10 = 36/38 Hit@1 (misses b1-020, b2-011).
Pass: union pool 10 + Jev v1 scores at least 36. Improvement: 37 or more. Below 36: regression, questions named.
Run: `python3 experiments/recall/e2e_union.py 10` (embeddings come from the cache, Jev is a cloud call).

### Result of step 1c (results/e2e-union-10.json)
Union pool 10 + Jev v1, 38 original questions: **Hit@1 38/38, Hit@5 38/38, MRR 1.000** (baseline 36/38, 0.947). Band: improvement.
Gains: b1-020 and b2-011, the two questions that were not in the embedding pool. Nothing lost.
Hard set v1 with the same setup (11 questions): Jev ranks the gold page first for 9 of 11; h-009 and h-011 are in no pool.
Cost: 8 fresh Jev calls, 372 served from the response cache (most union pools overlap pools already scored), $0.0003.
Caveats, in order of weight:
- A perfect score on a set I have iterated on is as much a sign of set saturation as of quality: 28 of the 38 were drafted from the chunks searched, and the two gains are the questions that motivated the merge.
- The hard-set questions that were not used to design the merge (h-008, h-010) both succeed, which is the better evidence, but n is 2.
- The cache holds one Jev answer per (question, passage) pair: run-to-run variation of Jev is not measured.
- The two hard questions that remain (h-009, h-011) are in no pool: the union does not fix cross-language misses.

## Step 2: French rewrites of h-008..h-011 (saved as h-012..h-015, same gold pages)
Prediction written before the run: if the cause is cross-language, at least 3 of the 4 French versions have the gold page in the union pool of 10,
and h-009 and h-011 move into the pool. If h-013 and h-015 stay out of the pool, the cause is something else (chunking, tables, ranking).
Confound I cannot remove: a faithful French translation uses the page's own words, so a gain means "French helps", not "language alone".

### Result of step 2 (union merge, pool 10 unless stated)
| English | French | Embedding rank EN -> FR | BM25 rank EN -> FR | Gold in pool 10 EN -> FR | Jev rank of gold (FR) |
|---|---|---|---|---|---|
| h-008 | h-012 | miss -> miss | 6 -> 1 | yes -> yes | 1 (0.97) |
| h-009 | h-013 | 14 -> 1 | miss -> 1 | no -> yes | 1 (0.96) |
| h-010 | h-014 | 4 -> 2 | miss -> 2 | yes -> yes | 1 (0.79) |
| h-011 | h-015 | miss -> 9 | miss -> 8 | no -> no (yes at pool 20) | 1 and 2, both gold pages (0.98 each) at pool 20 |

Against the prediction: 3 of 4 French versions are in the pool at 10 (met), but English was already 2 of 4, so the change is one question (h-009 to h-013).
h-009 moved in; h-011 did not at pool 10 (the fixed prediction said both would): it is found by both methods at page rank 8 to 9, which the round-robin places beyond chunk 10, and it is caught at pool 20.
Supported: French wording improves the embedding rank where the page is prose (h-009 14 to 1, h-010 4 to 2, h-011 miss to 9) and BM25 starts matching.
Not supported: h-012 (the B7 table) is still missed by the embedding in French, so for table rows the embedding itself, not the language, is the problem; BM25 rescues it.
Confound unchanged: the French wording also shares the page's own vocabulary, so this does not separate "language" from "word overlap".
Practical reading: a French-speaking analyst asking in French gets better retrieval than the English-question tests suggested; quarq's gold set is entirely English, so it understates French-user quality and overstates English-on-French-page quality.

## Update 2026-10-04: hard set reviewed
Marc checked every answer and page of h-003..h-011 against the PDFs (the PDF viewer's title-bar page index, not the printed page number, which differs by one in some of these files).
The four French rewrites (h-012..h-015) share those answers and pages and were checked through their English originals; their French wording was not separately reviewed.
All 15 rows now carry `synthetic-draft+human-accept`, so `quarq eval --dataset experiments/recall/hard_v1.jsonl` accepts the file (not run here: it would load two models on the GPU).
The earlier "not human-reviewed" caveats in the results above describe the state at the time. The LLM still chose the questions, so selection bias from drafting remains.
