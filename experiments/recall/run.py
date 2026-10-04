"""Pool recall for embedding / BM25 / hybrid, then end-to-end with the re-rankers."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.jev_rerank.client import JevClient  # noqa: E402
from experiments.jev_rerank.jev_reranker import JevReranker  # noqa: E402
from experiments.recall.hybrid import BM25, Candidates, HybridRetriever  # noqa: E402
from quarq.config import load_config  # noqa: E402
from quarq.eval.dataset import load_gold  # noqa: E402
from quarq.eval.runner import run_eval  # noqa: E402
from quarq.rag.embedder import Embedder  # noqa: E402
from quarq.rag.reranker import Reranker  # noqa: E402
from quarq.rag.store import VectorStore  # noqa: E402

GOLD = ROOT / "quarq" / "eval" / "datasets" / "quarq_gold_v1.jsonl"


def in_pool(chunks, gold, n):
    return any((c.source, int(c.page)) in gold for c in chunks[:n])


def main() -> None:
    cfg = load_config()
    store, embedder = VectorStore(cfg), Embedder(model_name=cfg.embedder.model)
    gold = load_gold(GOLD)
    n_chunks = store.count()
    cands = Candidates(store, embedder, BM25(store.list_chunks()))
    out: dict = {"generated_at": datetime.now(UTC).isoformat(), "corpus_chunks": n_chunks, "pool": {}}

    found: dict[str, dict[int, set[str]]] = {m: {10: set(), 20: set()} for m in ("embedding", "keyword", "hybrid")}
    for g in gold:
        want = {(r.source, r.page) for r in g.gold}
        for m in found:
            pool = getattr(cands, m)(g.question)
            for n in (10, 20):
                if in_pool(pool, want, n):
                    found[m][n].add(g.id)
    for n in (10, 20):
        e, h = found["embedding"][n], found["hybrid"][n]
        row = {m: len(found[m][n]) for m in found}
        row["hybrid_gained"] = sorted(h - e)
        row["hybrid_lost"] = sorted(e - h)
        out["pool"][f"N={n}"] = row
        print(f"pool N={n}:", json.dumps(row))
    out["not_in_any_hybrid_pool20"] = sorted({g.id for g in gold} - found["hybrid"][20])
    print("never in hybrid pool 20:", out["not_in_any_hybrid_pool20"])

    client = JevClient()
    out["end_to_end"] = {}
    arms = {
        "hybrid10+jev": lambda: HybridRetriever(cands, cfg, JevReranker(client), 10),
        "hybrid20+jev": lambda: HybridRetriever(cands, cfg, JevReranker(client), 20),
        "hybrid10+bge": lambda: HybridRetriever(
            cands, cfg, Reranker(cfg.rag.reranker_model, max_length=cfg.rag.rerank_max_length), 10),
    }
    for name, make in arms.items():
        res = run_eval(make(), gold, cfg, dataset_name="quarq_gold_v1", corpus_chunk_count=n_chunks)
        h1 = {q.id: q.metrics["hit@1"] for q in res.per_question}
        out["end_to_end"][name] = {"hit1": int(sum(h1.values())), "hit5": round(res.aggregate["hit@5"] * len(gold)),
                                   "mrr": round(res.aggregate["mrr"], 4),
                                   "misses_hit1": [q for q, v in h1.items() if v == 0]}
        print(name, out["end_to_end"][name])
    out["jev_total_cost_usd"] = round(client.cost_usd, 4)
    path = ROOT / "experiments" / "recall" / "results" / f"run-{datetime.now(UTC):%Y%m%dT%H%M%S}.json"
    path.write_text(json.dumps(out, indent=2))
    print("jev cost", out["jev_total_cost_usd"], "wrote", path)


if __name__ == "__main__":
    main()
