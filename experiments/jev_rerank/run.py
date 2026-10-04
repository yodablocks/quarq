"""Run the gold set through three retrieval arms and write one JSON result."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.jev_rerank.client import JevClient  # noqa: E402
from experiments.jev_rerank.jev_reranker import JevReranker  # noqa: E402
from quarq.config import load_config  # noqa: E402
from quarq.eval.dataset import load_gold  # noqa: E402
from quarq.eval.runner import run_eval  # noqa: E402
from quarq.rag.embedder import Embedder  # noqa: E402
from quarq.rag.reranker import Reranker  # noqa: E402
from quarq.rag.retriever import Retriever  # noqa: E402
from quarq.rag.store import VectorStore  # noqa: E402

GOLD = ROOT / "quarq" / "eval" / "datasets" / "quarq_gold_v1.jsonl"


class Timed:
    """Wraps a retriever and records wall time per query."""

    def __init__(self, inner: Retriever) -> None:
        self.inner = inner
        self.ms: list[float] = []

    def retrieve(self, *a, **kw):
        t = time.time()
        out = self.inner.retrieve(*a, **kw)
        self.ms.append((time.time() - t) * 1000)
        return out


def pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="embedding,bge,jev")
    args = ap.parse_args()

    cfg = load_config()
    store, embedder = VectorStore(cfg), Embedder(model_name=cfg.embedder.model)
    gold = load_gold(GOLD)
    n_chunks = store.count()
    client = JevClient()
    out: dict = {"generated_at": datetime.now(UTC).isoformat(), "n_questions": len(gold),
                 "corpus_chunks": n_chunks, "rerank_top_n": cfg.rag.rerank_top_n, "arms": {}}

    for arm in args.arms.split(","):
        if arm == "embedding":
            reranker = None
        elif arm == "bge":
            reranker = Reranker(cfg.rag.reranker_model, max_length=cfg.rag.rerank_max_length)
        else:  # "jev" or "jev:v2"
            reranker = JevReranker(client, variant=arm.partition(":")[2] or "v1")
        timed = Timed(Retriever(store, embedder, cfg, reranker=reranker))
        res = run_eval(timed, gold, cfg, dataset_name="quarq_gold_v1", corpus_chunk_count=n_chunks)
        entry = {"aggregate": res.aggregate,
                 "per_question_hit1": {q.id: q.metrics["hit@1"] for q in res.per_question},
                 "query_ms_p50": statistics.median(timed.ms), "query_ms_p95": pct(timed.ms, 0.95)}
        if arm.startswith("jev"):
            entry.update(ties_at_top=reranker.ties_at_top, fresh_calls=client.fresh_calls,
                         cache_hits=client.cache_hits, input_tokens=client.input_tokens,
                         cost_usd=round(client.cost_usd, 5),
                         call_ms_p50=statistics.median(client.latencies_ms) if client.latencies_ms else None,
                         call_ms_p95=pct(client.latencies_ms, 0.95) if client.latencies_ms else None,
                         rerank_wall_ms_p50=statistics.median(reranker.calls_wall_ms))
        out["arms"][arm] = entry
        print(arm, json.dumps(entry["aggregate"]), f"p50={entry['query_ms_p50']:.0f}ms")

    path = ROOT / "experiments" / "jev_rerank" / "results" / f"run-{datetime.now(UTC):%Y%m%dT%H%M%S}.json"
    path.write_text(json.dumps(out, indent=2))
    print("wrote", path)


if __name__ == "__main__":
    main()
