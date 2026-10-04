"""bge re-ranker on the union pool, over the 38 gold questions and the hard set. Paced."""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "2")

import json  # noqa: E402
import statistics  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.recall import probe  # noqa: E402
from experiments.recall.e2e_union import CachedEmbedder  # noqa: E402
from experiments.recall.hybrid import Candidates, HybridRetriever  # noqa: E402
from quarq.config import load_config  # noqa: E402
from quarq.eval.dataset import load_gold  # noqa: E402
from quarq.eval.runner import run_eval  # noqa: E402
from quarq.rag.reranker import Reranker  # noqa: E402
from quarq.rag.store import VectorStore  # noqa: E402

PAUSE_S = 3.0


class Paced:
    def __init__(self, inner) -> None:
        self.inner, self.ms = inner, []

    def retrieve(self, *a, **kw):
        t = time.time()
        out = self.inner.retrieve(*a, **kw)
        self.ms.append((time.time() - t) * 1000)
        time.sleep(PAUSE_S)
        return out


def main() -> None:
    cfg = load_config()
    store = VectorStore(cfg)
    cands = Candidates(store, CachedEmbedder(cfg), probe.bm25_index(store))
    bge = Reranker(cfg.rag.reranker_model, max_length=cfg.rag.rerank_max_length)
    sets = {"gold38": ROOT / "quarq" / "eval" / "datasets" / "quarq_gold_v1.jsonl",
            "hard15": ROOT / "experiments" / "recall" / "hard_v1.jsonl"}
    out = {}
    for name, path in sets.items():
        gold = load_gold(path, require_accepted=False)
        retr = Paced(HybridRetriever(cands, cfg, bge, 10, merge="union"))
        res = run_eval(retr, gold, cfg, dataset_name=name, corpus_chunk_count=store.count())
        h1 = {q.id: q.metrics["hit@1"] for q in res.per_question}
        out[name] = {"n": len(gold), "hit1": int(sum(h1.values())), "hit5": round(res.aggregate["hit@5"] * len(gold)),
                     "mrr": round(res.aggregate["mrr"], 4), "misses_hit1": [q for q, v in h1.items() if v == 0],
                     "query_ms_p50": round(statistics.median(retr.ms))}
        print(name, json.dumps(out[name]), flush=True)
    (ROOT / "experiments" / "local_rerank" / "results" / "bge-union.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
