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
