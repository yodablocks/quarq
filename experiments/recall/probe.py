"""Probe ONE question: where does its gold page land, and does Jev put it first?

Light on the machine on purpose: one question per run, thread count capped, the query
embedding and the BM25 index cached on disk (a repeat probe loads no model), Jev is a cloud
call (no local GPU), and the bge reranker is never loaded.

  python3 experiments/recall/probe.py "question text" --gold source.pdf:page [--gold ...]
  python3 experiments/recall/probe.py --id h-001            # re-run a saved hard question
  ... --save h-001                                          # append it to hard_v1.jsonl
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import pickle  # noqa: E402
import statistics  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.jev_rerank.client import JevClient  # noqa: E402
from experiments.jev_rerank.jev_reranker import JevReranker  # noqa: E402
from experiments.recall.hybrid import BM25, fuse, interleave  # noqa: E402
from quarq.config import load_config  # noqa: E402
from quarq.rag.store import VectorStore  # noqa: E402

HERE = Path(__file__).resolve().parent
CACHE = HERE / ".cache"
HARD = HERE / "hard_v1.jsonl"
DEPTH = 40


def parse_gold(raw: list[str]) -> set[tuple[str, int]]:
    out = set()
    for item in raw:
        source, _, page = item.rpartition(":")
        if not source or not page.isdigit():
            sys.exit(f"--gold must look like file.pdf:12, got {item!r}")
        out.add((source, int(page)))
    return out


def load_saved(qid: str) -> tuple[str, set[tuple[str, int]]]:
    for line in HARD.read_text().splitlines() if HARD.exists() else []:
        row = json.loads(line)
        if row["id"] == qid:
            return row["question"], {(g["source"], g["page"]) for g in row["gold"]}
    sys.exit(f"no saved question {qid!r} in {HARD.name}")


_EMBEDDER = None


def query_embedding(cfg, question: str) -> list[float]:
    global _EMBEDDER
    CACHE.mkdir(exist_ok=True)
    path = CACHE / "query_emb.json"
    cache = json.loads(path.read_text()) if path.exists() else {}
    key = hashlib.sha256(f"{cfg.embedder.model}\n{question}".encode()).hexdigest()
    if key not in cache:
        from quarq.rag.embedder import Embedder

        if _EMBEDDER is None:
            print("(loading the embedding model, once per process)")
            _EMBEDDER = Embedder(model_name=cfg.embedder.model)
        cache[key] = _EMBEDDER.embed_query(question)
        path.write_text(json.dumps(cache))
    return cache[key]


def bm25_index(store: VectorStore) -> BM25:
    CACHE.mkdir(exist_ok=True)
    path, count = CACHE / "bm25.pkl", store.count()
    if path.exists():
        saved_count, index = pickle.loads(path.read_bytes())
        if saved_count == count:
            return index
    print(f"(building the BM25 index over {count} chunks, cached afterwards)")
    index = BM25(store.list_chunks())
    path.write_bytes(pickle.dumps((count, index)))
    return index


def page_ranks(chunks, gold) -> str:
    seen: list[tuple[str, int]] = []
    for c in chunks:
        ref = (c.source, int(c.page))
        if ref not in seen:
            seen.append(ref)
    ranks = [f"p{i}" for i, ref in enumerate(seen, 1) if ref in gold]
    return ", ".join(ranks) if ranks else f"not in top {len(chunks)}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("question", nargs="?")
    ap.add_argument("--gold", action="append", default=[], help="source.pdf:page (repeatable)")
    ap.add_argument("--id", help="re-run a question saved in hard_v1.jsonl")
    ap.add_argument("--pool", type=int, default=10, help="chunks handed to Jev (default 10, max 20)")
    ap.add_argument("--merge", choices=("rrf", "union"), default="rrf",
                    help="how embedding and BM25 candidates are merged (default rrf)")
    ap.add_argument("--no-jev", action="store_true", help="skip the cloud call (free)")
    ap.add_argument("--save", metavar="ID", help="append this question to hard_v1.jsonl")
    ap.add_argument("--provenance", default="hand-written",
                    help="provenance stored with --save (default hand-written; "
                         "e.g. synthetic-draft, synthetic-draft+human-accept)")
    ap.add_argument("--note", default="", help="note stored with --save")
    args = ap.parse_args()
    if args.id:
        question, gold = load_saved(args.id)
    elif args.question and args.gold:
        question, gold = args.question, parse_gold(args.gold)
    else:
        ap.error("give a question and at least one --gold, or --id")
    pool_n = min(args.pool, 20)

    cfg = load_config()
    store = VectorStore(cfg)
    index = bm25_index(store)
    known = {(c.source, int(c.page)) for c in index.chunks}
    missing = sorted(gold - known)
    if missing:
        sys.exit(f"gold page not in the corpus: {missing} (pages use the PDF viewer index, 1 = first page)")

    emb = sorted(store.query(query_embedding(cfg, question), k=DEPTH, filters=None),
                 key=lambda c: c.similarity, reverse=True)
    kw = index.top(question, DEPTH)
    hybrid = fuse([emb, kw]) if args.merge == "rrf" else interleave([emb, kw])
    print(f"\nQ: {question}\ngold: {sorted(gold)}\n")
    print("gold page rank (distinct pages, best first):")
    for name, pool in (("embedding", emb), ("BM25", kw), (f"hybrid/{args.merge}", hybrid)):
        print(f"  {name:<10} {page_ranks(pool, gold)}")
    in_pool = any((c.source, int(c.page)) in gold for c in hybrid[:pool_n])
    print(f"  hybrid pool of {pool_n} chunks contains the gold page: {'yes' if in_pool else 'NO'}")

    if not args.no_jev:
        client = JevClient()
        reranked = JevReranker(client).rerank(question, hybrid[:pool_n])
        print(f"\nJev over the hybrid pool of {pool_n} (top 5 pages):")
        shown: set[tuple[str, int]] = set()
        for c in reranked:
            ref = (c.source, int(c.page))
            if ref in shown:
                continue
            shown.add(ref)
            mark = "  <== gold" if ref in gold else ""
            if len(shown) <= 5 or mark:
                print(f"  {len(shown)}. {c.rerank_score:.3f}  {c.source} p{c.page}{mark}")
        print(f"cost ${client.cost_usd:.5f}, {client.fresh_calls} fresh calls "
              f"({client.cache_hits} cached)"
              + (f", call p50 {statistics.median(client.latencies_ms):.0f} ms" if client.latencies_ms else ""))

    if args.save:
        if HARD.exists() and any(json.loads(l)["id"] == args.save for l in HARD.read_text().splitlines()):
            sys.exit(f"id {args.save!r} already in {HARD.name}")
        row = {"id": args.save, "question": question, "doc_type": None, "provenance": args.provenance,
               "gold": [{"source": s, "page": p} for s, p in sorted(gold)], "note": args.note}
        with HARD.open("a") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"\nsaved {args.save} to {HARD.name}")


if __name__ == "__main__":
    main()
