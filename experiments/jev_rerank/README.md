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
