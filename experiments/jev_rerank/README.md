# Experiment 1: Jev as the re-ranker

Question: on quarq's own French/English regulatory corpus and 38-question gold set, does TypeSafe's
Jev (a Noul per question + passage pair) re-rank as well as the local `bge-reranker-v2-m3`
cross-encoder, and at what latency and cost?

Part of note 001 in the from_Q2P repo. Throwaway code: it lives outside the `quarq` package on
purpose, and `jev-latest` is named here because the repo rule bans model names inside `rag/` and `llm/`.

## Arms (same 38 questions, same corpus, same date-aware step)
1. `embedding`: no re-ranker.
2. `bge`: the production cross-encoder on the top 10.
3. `jev`: one Noul per (question, chunk) pair on the same top 10, ordered by probability.
   Ties keep the embedding order (stable sort) and are counted.

## Decision bands, written before the first run
Baseline is `bge` on this run (README says 33/38 Hit@1; the run's own number wins).
- Jev **matches**: Hit@1 within 2 questions of `bge`. (One question is 2.6 points; this is noise.)
- Jev **beats**: Hit@1 at least 3 questions above `bge`.
- Jev **loses**: Hit@1 at least 3 questions below `bge`.
Latency and cost are reported, not gated. Nothing here proves anything about unseen questions.

## Known limits
- 28 of the 38 questions were drafted by an LLM from the chunks being searched.
- Cloud API: the question and the passage text leave the machine. Corpus is public documents.
- Jev reads English instructions over French passages. That is part of what is being tested.

## Run
```bash
export TYPESAFE_API_KEY=...        # already set in Marc's shell
python3 experiments/jev_rerank/run.py --arms embedding,bge,jev
```
Responses are cached in `.cache/`, so a re-run costs nothing. Results go to `results/`.

## Result (run 2026-10-04, one run, results/run-20261004T032932.json)
| Arm | Hit@1 | Hit@3 | Hit@5 | MRR | Query latency p50 |
|---|---|---|---|---|---|
| embedding | 23/38 | 32/38 | 35/38 | 0.74 | 50 ms |
| bge (production) | 33/38 | 35/38 | 36/38 | 0.90 | 2,079 ms (Apple GPU) |
| jev | 36/38 | 36/38 | 36/38 | 0.95 | 862 ms (12 parallel calls) |

Band: **beats**, but by exactly 3 questions, the threshold. Jev fixed b2-010, b2-017 and y-09 and broke none.
Both arms miss b1-020 and b2-011: Hit@5 is identical (36/38), so the shortlist, not the re-ranker, is the ceiling.
Jev: 380 calls, 338,274 input tokens, $0.014, call latency p50 524 ms and p95 805 ms, 2 queries with a tie at the top.
Not measured: repeat-run stability, wording sensitivity, French-only questions separately, any prompt variant (one wording, not tuned).

## Follow-up (a): wording sensitivity, bands written before the run
Same arm, same 38 questions, three wordings of the Noul question (v1 original, v2 shorter, v3 "reader with only this passage").
The `bge` baseline is 33/38 from the first run. Per-wording Hit@1, then:
- **Robust win**: all three wordings at least 35/38.
- **Holds**: all three at least 33/38.
- **Unstable**: any two wordings differ by 3 or more questions, or any wording below 33/38. Then the first result was a wording effect.
Wordings were written once, before running, and are not tuned afterwards.
Repeat-run noise of one fixed wording is NOT tested: responses are cached, so a rerun returns the same answers.
Run: `python3 experiments/jev_rerank/run.py --arms jev:v2,jev:v3`

### Result of follow-up (a) (results/run-20261004T033321.json)
| Wording | Hit@1 | Hit@5 | MRR | Misses |
|---|---|---|---|---|
| v1 (original) | 36/38 | 36/38 | 0.947 | b1-020, b2-011 |
| v2 (shorter) | 36/38 | 36/38 | 0.947 | b1-020, b2-011 |
| v3 (reader with only this passage) | 36/38 | 36/38 | 0.947 | b1-020, b2-011 |

Band: **robust win** (all three at least 35/38, spread 0). Per-question outcomes were identical across wordings.
The two misses are first-stage failures: for both, the gold page is not in the embedding top 40 (checked directly), so no re-ranker could reach it. b1-020 is an exact-date factsheet question about BNP's index weight; b2-011 is an ECB FSR page.
Latency again: call p50 about 520 ms, p95 about 820 ms; each wording about $0.014.
Correction (code review, 2026-10-05): `run.py` shared one Jev client across arms, so in `results/run-20261004T033321.json` the `jev:v3` fields `fresh_calls`, `cost_usd`, `call_ms_p50` and `call_ms_p95` are cumulative over v2 and v3 (760 calls, $0.0278). The per-wording figures above (about 380 calls and $0.014 each) are the right ones. The file is left as recorded; `run.py` now creates a client per arm.
Still not tested: rerunning one fixed wording (cache returns the same answers), French-only questions, harder or larger gold sets, local alternatives.
Identical results across three wordings is itself a mild warning: this gold set may be too easy to separate them (28 of 38 questions were drafted from the chunks themselves).

## Experiment (a): repeat-run variation of Jev, bands written before the run
Every earlier Jev result came from a cache that returns one answer per (question, passage) pair, so run-to-run variation was never measured.
`repeat.py` scores the same pairs three times: pass 0 from the cache (the earlier answers), passes 1 and 2 with the cache bypassed (`fresh=True`, nothing written).
Same union pool of 10, wording v1, 38 gold + 15 hard questions (about 530 pairs per pass). Cloud calls only, about $0.03, no local model.
Bands (on the questions' Hit@1 and top-1 page across the three passes):
- **Stable**: every pass gives the same Hit@1 on both sets and no question's top-1 page changes.
- **Some variation**: scores differ (spread over 0.05 on any pair) but Hit@1 is the same in every pass and at most 2 questions change top-1.
- **Unstable**: Hit@1 differs by 2 or more questions between passes on either set, or more than 2 questions change top-1.
Also reported: pairs with identical scores, max and mean score spread. If Jev is deterministic at the API level the result is "stable, identical": that is a legitimate outcome.
Run: `python3 experiments/jev_rerank/repeat.py`

### Result of experiment (a) (results/repeat.json)
Jev is **not deterministic**. Same pairs, same wording, three passes (one from the cache, two fresh), 490 distinct (question, passage) pairs:
- 281 pairs (57%) returned identical scores in all three passes; mean spread 0.010; 24 pairs (5%) spread by more than 0.05; one pair by 0.25.
- Hit@1 per pass: gold38 **38, 38, 37**; hard15 **12, 12, 12**.
- Two gold questions (b1-005, y-08) changed their top-1 page in at least one pass; none of the hard questions did. In the third pass one of those lost the gold page at rank 1.
- Cost of the two fresh passes: 1,060 calls (530 per pass, as printed by the run; `repeat.json` does not store the call count), 996,628 input tokens in total, $0.042.

Against the bands written first: not **Stable** (Hit@1 differs between passes) and not **Unstable** (the gap is 1 question, and 2 questions changed top-1); the bands left a gap
for "Hit@1 differs by exactly 1", which is what happened. Plain reading: **mild run-to-run variation**; the 38/38 should be read as 37 to 38.
The earlier comparisons still hold in direction: bge is deterministic at 33/38 on the embedding pool and 33/38 on the union pool, and Jev has not been seen below 37 on the 38.
Not saved by this run: the per-pass top-1 pages, so which question lost the gold page in pass 2 is not recorded (the script only keeps the questions whose top-1 changed).
Not tested: variation of the wording variants (v2, v3), the embedding pool, or more than three passes (a 4th and 5th pass would tighten the estimate).
