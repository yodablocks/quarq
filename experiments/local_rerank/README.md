# Experiment 3: a local re-ranker on the union pool

Privacy follow-up to experiments/jev_rerank and experiments/recall. Jev is a cloud API, which breaks quarq's
"local handling" promise. The production re-ranker bge-reranker-v2-m3 is already local. Known so far:
- bge on the embedding pool of 10: 33/38 Hit@1 (experiments/jev_rerank).
- bge on the RRF pool of 10: 32/38.
- Jev on the union pool of 10: 38/38 (experiments/recall).
Question: does the union pool lift bge too? Unknown, and it decides whether a fully local path can get close to Jev.

## Step 3a (no new model, no download): bge on the union pool of 10
Arms: bge + union pool, on the 38 original questions, and on the 15-question hard set (h-001..h-015).
Pacing: 3 s pause after every question to keep the GPU cool. Query embeddings come from the cache; BM25 index too.

Bands, written before the run (baseline bge 33/38 on the embedding pool):
- **Lifts**: bge + union reaches at least 35/38.
- **Matches Jev**: at least 37/38.
- **No lift**: 34 or fewer, or any question lost that bge had right on the embedding pool (named by id, if the per-question data allows).
Jev reference on the hard set (h-001..h-011): gold ranked first for 9 of 11, h-009 and h-011 not in the pool.
Latency is reported per query (p50), not gated.

## Not covered here
- Other local re-rankers (would need a download: ask first).
- A local LLM as a pointwise judge via LM Studio (heavier on the machine; your LM Studio is not on this host).

## Run
`python3 experiments/local_rerank/run.py` (loads bge once; ctrl-C stops it)

## Result of step 3a (results/bge-union.json, one run)
| Set | Reranker + pool | Hit@1 | Hit@5 | MRR | Query p50 |
|---|---|---|---|---|---|
| gold38 | bge, embedding pool (earlier run) | 33 | 36 | 0.901 | 2,079 ms |
| gold38 | bge, union pool | 33 | 38 | 0.926 | 2,099 ms |
| gold38 | Jev, union pool (experiments/recall) | 38 | 38 | 1.000 | 862 ms |
| hard15 | bge, union pool | 10 | 12 | 0.733 | 2,049 ms |
| hard15 | Jev, union pool 10 (from experiments/recall step 1c and step 2; Jev on h-015 not run at pool 10 because the gold page is not in the pool) | 12 | n/a | n/a | about 850 ms |

Band: **no lift** at Hit@1 (33, not 35). The union gets the gold page into bge's pool (Hit@5 36 to 38, MRR 0.901 to 0.926), but bge does not rank it first:
its Hit@1 misses on the 38 are the same five as before (b1-020, b2-010, b2-011, b2-017, y-09), including the two the union added to the pool.
On the hard set bge misses h-001, h-002, h-009, h-011 and h-015 at rank 1; h-009, h-011 and h-015 are not in the pool at 10 for any re-ranker, so the real difference with Jev is h-001 and h-002 (Jev first, bge not).
Gap to Jev at Hit@1: 5 questions on the 38, 2 on the hard set. bge was about 2.4 times slower per query on this machine (local GPU against a network call).
Reading, not tested: the Jev question asks whether the passage states the answer; bge scores topical relevance. The prompt wording, not only the model, may carry the difference.
Caveats: one run, same small sets, 3 s pauses between questions (latency excludes them).

## Step 3b: a local LLM as a pointwise judge (gemma-4-12b-qat on the LAN LM Studio host)
Question: can a local model, asked whether the passage states the answer, re-rank as well as Jev, and does the wording matter as much as the model?
Score: P(yes) from the first token's log-probabilities (yes/no variants, normalized). Same union pool of 10, same two sets (38 gold, 15 hard).
Two wordings, same model, committed in `llm_judge.py`:
- **A** "does the passage state the answer" (the Jev wording).
- **B** "is the passage relevant to the question" (topical, what bge is trained for).
Requests are sequential, responses cached on disk. The question and passage go to the LM Studio host on the LAN only.

Bands, written before the run. References on the 38: bge 33, Jev 38.
- **Matches bge**: wording A scores 31 to 35. **Closes the gap**: 36 or more. **Worse than bge**: 30 or fewer.
- **Wording matters**: A and B differ by 3 or more questions on the 38. Otherwise the wording is not carrying the result at this sample size.
- A run with more than 5% of requests returning no yes/no signal is void (reported as such).
Timing first: the first 10 requests are timed alone; if a request takes over 3 s on average the full run (about 530 per wording) is not started without asking.
Run: `python3 experiments/local_rerank/llm_judge.py --wording A --limit 10` (timing), then without `--limit`.

### Heat (found after the timing test)
The LM Studio host (192.168.1.107) is the same Mac that runs everything else, so the 12B model's inference heats this machine.
The timing test measured 6.4 s per request (max 7.7 s) with the model running continuously.
`llm_judge.py` therefore takes `--burst N --cool S`: after every N fresh requests it sleeps S seconds. Responses are cached, so any stop resumes cleanly.
It also reads `pmset -g therm`; if macOS reports a CPU speed limit below 100 it pauses 180 s and stops if still throttled. That is a late tripwire
(it fires once throttling has begun), not a temperature reading, so the duty cycle is the real protection. Temperature is not measured here.
`--thermal-log FILE` adds a better guard: run `sudo powermetrics --samplers thermal -i 5000 > FILE` in a second terminal and the judge reads the latest
"Current pressure level" before every request, waiting 60 s while it is above Nominal (stops after 10 min), and stops if the file is stale or unreadable.
This is a pressure level (Nominal, Moderate, Heavy, ...), not degrees; how it maps to temperature on this chip is unknown.

## Outcome of step 3b: not completed, no result
The judge run was stopped by hand after 26 of the 150 requests the hard set needs (10 of them for the first gold question), because the Mac overheated:
a continuous run reached the temperature the user reported (about 110 C within a minute), and even a 4-request burst raised macOS thermal pressure to Moderate.
No Hit@1 was produced, so **the wording hypothesis (does "states the answer" beat "is relevant") is untested** and nothing here says a local LLM judge works or fails.
What is recorded: a 12B model on this machine took 6.4 s per request (max 7.7 s) on the 10-request timing test, and the host at 192.168.1.107 is the same Mac.
The code stays so the run can be repeated on a machine that can take it, or with a smaller model after committing new bands. Cached scores are in `.cache/` (gitignored).
The conclusion from step 3a stands: bge on the union pool is about 5 questions behind Jev on the 38.
