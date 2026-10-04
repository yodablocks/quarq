"""Experiment (a): how much do Jev's answers vary between runs? Cloud calls only, no local model.

Pass 0 reads the cached answers from the earlier runs; passes 1 and 2 bypass the cache and ask again.
Same union pool of 10, same question wording (v1), over the 38 gold and 15 hard questions.
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.jev_rerank.client import JevClient  # noqa: E402
from experiments.jev_rerank.jev_reranker import JevReranker  # noqa: E402
from experiments.recall import probe  # noqa: E402
from experiments.recall.e2e_union import CachedEmbedder  # noqa: E402
from experiments.recall.hybrid import Candidates, HybridRetriever  # noqa: E402
from quarq.config import load_config  # noqa: E402
from quarq.eval.dataset import load_gold  # noqa: E402
from quarq.eval.runner import run_eval  # noqa: E402
from quarq.rag.store import VectorStore  # noqa: E402

PASSES = 3
SETS = {"gold38": ROOT / "quarq" / "eval" / "datasets" / "quarq_gold_v1.jsonl",
        "hard15": ROOT / "experiments" / "recall" / "hard_v1.jsonl"}


def main() -> None:
    cfg = load_config()
    store = VectorStore(cfg)
    cands = Candidates(store, CachedEmbedder(cfg), probe.bm25_index(store))
    client = JevClient()
    scores: dict[tuple[str, str], list[float]] = defaultdict(list)
    per_pass: list[dict] = []
    for n in range(PASSES):
        calls_before = client.fresh_calls
        row: dict = {"pass": n, "fresh": n > 0}
        for name, path in SETS.items():
            reranker = JevReranker(client, fresh=n > 0)
            res = run_eval(HybridRetriever(cands, cfg, reranker, 10, merge="union"), load_gold(path, require_accepted=False),
                           cfg, dataset_name=name, corpus_chunk_count=store.count())
            h1 = {q.id: q.metrics["hit@1"] for q in res.per_question}
            row[name] = {"hit1": int(sum(h1.values())), "top1": {q.id: list(q.retrieved[0]) if q.retrieved else None for q in res.per_question}}
            for q, c, s in reranker.record:
                scores[(q, c)].append(s)
        row["fresh_calls"] = client.fresh_calls - calls_before
        per_pass.append(row)
        print(f"pass {n}:", {k: row[k]["hit1"] for k in SETS}, "fresh calls", row["fresh_calls"], flush=True)

    pairs = [v for v in scores.values() if len(v) == PASSES]
    spread = [max(v) - min(v) for v in pairs]
    out = {"passes": PASSES, "pairs": len(pairs), "pairs_identical": sum(1 for d in spread if d == 0),
           "max_spread": round(max(spread), 4) if spread else None,
           "mean_spread": round(statistics.mean(spread), 5) if spread else None,
           "pairs_spread_over_0.05": sum(1 for d in spread if d > 0.05),
           "pairs_spread_over_0.2": sum(1 for d in spread if d > 0.2),
           "hit1_per_pass": {k: [p[k]["hit1"] for p in per_pass] for k in SETS},
           "questions_top1_changed": {k: sorted(q for q in per_pass[0][k]["top1"]
                                                if len({json.dumps(p[k]["top1"][q]) for p in per_pass}) > 1) for k in SETS},
           "input_tokens": client.input_tokens, "cost_usd": round(client.cost_usd, 4)}
    print(json.dumps(out))
    (ROOT / "experiments" / "jev_rerank" / "results" / "repeat.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
