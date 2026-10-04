"""A local LLM (via LM Studio) as a pointwise re-ranker: P(yes) from the first token's log-probabilities.

Same candidate pool as the other experiments (union of embedding and BM25 chunks, first 10).
Requests are sequential, with responses cached on disk. Passage and question go to the LM Studio host
only (your LAN), never to a cloud API.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sqlite3  # noqa: E402
import statistics  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from dataclasses import replace  # noqa: E402
from pathlib import Path  # noqa: E402

import requests  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.recall import probe  # noqa: E402
from experiments.recall.e2e_union import CachedEmbedder  # noqa: E402
from experiments.recall.hybrid import Candidates, HybridRetriever  # noqa: E402
from quarq.config import load_config  # noqa: E402
from quarq.eval.dataset import load_gold  # noqa: E402
from quarq.eval.runner import run_eval  # noqa: E402
from quarq.rag.store import RetrievedChunk, VectorStore  # noqa: E402

CACHE = Path(__file__).resolve().parent / ".cache" / "judge.sqlite"

WORDINGS = {
    # A: the Jev v1 wording, rephrased as a one-word reply
    "A": ("Does the passage itself state the specific information that answers the question? "
          "Answer yes only if the answer is written in the passage. A passage that is about the same "
          "topic, edition or period but does not contain the answer is a no. Reply with exactly one word: yes or no."),
    # B: topical relevance, what a cross-encoder is trained to score
    "B": ("Is the passage relevant to the question? Answer yes if the passage is about the subject of "
          "the question, and no if it is about something else. Reply with exactly one word: yes or no."),
}


class Judge:
    def __init__(self, url: str, model: str, wording: str) -> None:
        self.url, self.model, self.wording = url.rstrip("/"), model, wording
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(CACHE)
        self.db.execute("CREATE TABLE IF NOT EXISTS r (key TEXT PRIMARY KEY, p_yes REAL, ms REAL)")
        self.fresh, self.hits, self.no_signal, self.ms = 0, 0, 0, []
        self.limit: int | None = None

    def score(self, question: str, passage: str) -> float:
        prompt = f"Question: {question}\n\nPassage: {passage}\n\n{WORDINGS[self.wording]}"
        key = hashlib.sha256(f"{self.model}\n{prompt}".encode()).hexdigest()
        row = self.db.execute("SELECT p_yes FROM r WHERE key=?", (key,)).fetchone()
        if row:
            self.hits += 1
            return row[0]
        if self.limit is not None and self.fresh >= self.limit:
            raise SystemExit(f"stopped after {self.fresh} fresh requests (--limit)")
        started = time.time()
        resp = requests.post(f"{self.url}/chat/completions", timeout=300, json={
            "model": self.model, "temperature": 0, "max_tokens": 1, "logprobs": True, "top_logprobs": 10,
            "messages": [{"role": "user", "content": prompt}]})
        resp.raise_for_status()
        ms = (time.time() - started) * 1000
        tops = resp.json()["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
        py = sum(math.exp(t["logprob"]) for t in tops if t["token"].strip().lower() == "yes")
        pn = sum(math.exp(t["logprob"]) for t in tops if t["token"].strip().lower() == "no")
        if py + pn == 0:
            self.no_signal += 1
            p = 0.5
        else:
            p = py / (py + pn)
        self.db.execute("INSERT OR REPLACE INTO r VALUES (?,?,?)", (key, p, ms))
        self.db.commit()
        self.fresh += 1
        self.ms.append(ms)
        return p


class JudgeReranker:
    def __init__(self, judge: Judge) -> None:
        self.judge, self.ties = judge, 0

    def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        scores = [self.judge.score(query, c.content) for c in chunks]
        if scores and sum(1 for s in scores if s == max(scores)) > 1:
            self.ties += 1
        scored = [replace(c, rerank_score=s) for c, s in zip(chunks, scores, strict=True)]
        return sorted(scored, key=lambda c: c.rerank_score or 0.0, reverse=True)  # stable


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wording", choices=sorted(WORDINGS), required=True)
    ap.add_argument("--model", default="google/gemma-4-12b-qat")
    ap.add_argument("--url", help="LM Studio base url, default quarq config lmstudio.url")
    ap.add_argument("--sets", default="gold38,hard15")
    ap.add_argument("--limit", type=int, help="stop after this many fresh requests (timing test)")
    args = ap.parse_args()

    cfg = load_config()
    store = VectorStore(cfg)
    cands = Candidates(store, CachedEmbedder(cfg), probe.bm25_index(store))
    judge = Judge(args.url or cfg.lmstudio.url, args.model, args.wording)
    judge.limit = args.limit
    reranker = JudgeReranker(judge)
    paths = {"gold38": ROOT / "quarq" / "eval" / "datasets" / "quarq_gold_v1.jsonl",
             "hard15": ROOT / "experiments" / "recall" / "hard_v1.jsonl"}
    out: dict = {"model": args.model, "wording": args.wording}
    for name in args.sets.split(","):
        gold = load_gold(paths[name], require_accepted=False)
        res = run_eval(HybridRetriever(cands, cfg, reranker, 10, merge="union"), gold, cfg,
                       dataset_name=name, corpus_chunk_count=store.count())
        h1 = {q.id: q.metrics["hit@1"] for q in res.per_question}
        out[name] = {"n": len(gold), "hit1": int(sum(h1.values())), "hit5": round(res.aggregate["hit@5"] * len(gold)),
                     "mrr": round(res.aggregate["mrr"], 4), "misses_hit1": [q for q, v in h1.items() if v == 0]}
        print(name, json.dumps(out[name]), flush=True)
    out.update(fresh_requests=judge.fresh, cache_hits=judge.hits, no_signal=judge.no_signal, ties_at_top=reranker.ties,
               request_ms_p50=round(statistics.median(judge.ms)) if judge.ms else None)
    print(json.dumps({k: out[k] for k in ("fresh_requests", "cache_hits", "no_signal", "ties_at_top", "request_ms_p50")}))
    (ROOT / "experiments" / "local_rerank" / "results" / f"judge-{args.wording}.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
