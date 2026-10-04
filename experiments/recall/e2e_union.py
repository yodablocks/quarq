"""End to end: union pool + Jev re-ranker over the 38 gold questions (embeddings from the cache)."""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "2")

import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.jev_rerank.client import JevClient  # noqa: E402
from experiments.jev_rerank.jev_reranker import JevReranker  # noqa: E402
from experiments.recall import probe  # noqa: E402
from experiments.recall.hybrid import Candidates, HybridRetriever  # noqa: E402
from quarq.config import load_config  # noqa: E402
from quarq.eval.dataset import load_gold  # noqa: E402
from quarq.eval.runner import run_eval  # noqa: E402
from quarq.rag.store import VectorStore  # noqa: E402


class CachedEmbedder:
    """embed_query served from probe's on-disk cache; loads no model."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg

    def embed_query(self, text: str) -> list[float]:
        return probe.query_embedding(self.cfg, text)


def main() -> None:
    pool_n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    cfg = load_config()
    store = VectorStore(cfg)
    gold = load_gold(ROOT / "quarq" / "eval" / "datasets" / "quarq_gold_v1.jsonl")
    cands = Candidates(store, CachedEmbedder(cfg), probe.bm25_index(store))
    client = JevClient()
    retr = HybridRetriever(cands, cfg, JevReranker(client), pool_n, merge="union")
    res = run_eval(retr, gold, cfg, dataset_name="quarq_gold_v1", corpus_chunk_count=store.count())
    h1 = {q.id: q.metrics["hit@1"] for q in res.per_question}
    out = {"pool": pool_n, "hit1": int(sum(h1.values())), "hit5": round(res.aggregate["hit@5"] * len(gold)),
           "mrr": round(res.aggregate["mrr"], 4), "misses_hit1": [q for q, v in h1.items() if v == 0],
           "fresh_calls": client.fresh_calls, "cache_hits": client.cache_hits,
           "cost_usd": round(client.cost_usd, 5)}
    print(json.dumps(out))
    (ROOT / "experiments" / "recall" / "results" / f"e2e-union-{pool_n}.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
