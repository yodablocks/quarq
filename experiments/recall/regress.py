"""Regression check: over the original 38 gold questions, does a merge lose gold pages
that the embedding pool already had? No Jev calls, no bge. One embedder per process."""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "2")

import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.recall import probe  # noqa: E402
from experiments.recall.hybrid import fuse, interleave  # noqa: E402
from quarq.config import load_config  # noqa: E402
from quarq.eval.dataset import load_gold  # noqa: E402
from quarq.rag.store import VectorStore  # noqa: E402


def has_gold(chunks, gold, n):
    return any((c.source, int(c.page)) in gold for c in chunks[:n])


def main() -> None:
    cfg = load_config()
    store = VectorStore(cfg)
    index = probe.bm25_index(store)
    items = load_gold(ROOT / "quarq" / "eval" / "datasets" / "quarq_gold_v1.jsonl")
    found: dict[int, dict[str, set[str]]] = {n: {m: set() for m in ("embedding", "bm25", "rrf", "union")} for n in (10, 20)}
    for g in items:
        gold = {(r.source, r.page) for r in g.gold}
        emb = sorted(store.query(probe.query_embedding(cfg, g.question), k=probe.DEPTH, filters=None),
                     key=lambda c: c.similarity, reverse=True)
        kw = index.top(g.question, probe.DEPTH)
        pools = {"embedding": emb, "bm25": kw, "rrf": fuse([emb, kw]), "union": interleave([emb, kw])}
        for n in (10, 20):
            for name, pool in pools.items():
                if has_gold(pool, gold, n):
                    found[n][name].add(g.id)
    out = {}
    for n in (10, 20):
        base = found[n]["embedding"]
        row = {m: len(found[n][m]) for m in found[n]}
        for m in ("rrf", "union"):
            row[f"{m}_lost"] = sorted(base - found[n][m])
            row[f"{m}_gained"] = sorted(found[n][m] - base)
        out[f"N={n}"] = row
        print(f"N={n}:", json.dumps(row))
    path = ROOT / "experiments" / "recall" / "results" / "regress-38.json"
    path.write_text(json.dumps(out, indent=2))
    print("wrote", path)


if __name__ == "__main__":
    main()
