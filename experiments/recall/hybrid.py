"""BM25 plus embedding candidates fused by reciprocal rank. Throwaway."""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import replace

from quarq.config import QuarqConfig
from quarq.rag.dates import exact_dates, prefer_covering
from quarq.rag.embedder import Embedder
from quarq.rag.retriever import RerankerLike, _best_chunk_per_page
from quarq.rag.store import RetrievedChunk, VectorStore

_TOKEN = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    folded = unicodedata.normalize("NFKD", text.lower())
    folded = "".join(ch for ch in folded if not unicodedata.combining(ch))
    return _TOKEN.findall(folded)


class BM25:
    def __init__(self, chunks: list[RetrievedChunk], k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks, self.k1, self.b = chunks, k1, b
        self.tf: list[Counter[str]] = [Counter(tokenize(c.content)) for c in chunks]
        self.len = [sum(t.values()) for t in self.tf]
        self.avg = sum(self.len) / max(len(chunks), 1)
        self.post: dict[str, list[int]] = defaultdict(list)
        for i, t in enumerate(self.tf):
            for term in t:
                self.post[term].append(i)
        n = len(chunks)
        self.idf = {t: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.post.items()}

    def top(self, query: str, k: int) -> list[RetrievedChunk]:
        scores: dict[int, float] = defaultdict(float)
        for term in set(tokenize(query)):
            if term not in self.post:
                continue
            for i in self.post[term]:
                f = self.tf[i][term]
                norm = f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg)
                scores[i] += self.idf[term] * f * (self.k1 + 1) / norm
        best = sorted(scores, key=lambda i: scores[i], reverse=True)[:k]
        return [self.chunks[i] for i in best]


def _key(c: RetrievedChunk) -> str:
    return str(c.metadata.get("chunk_id"))


def fuse(lists: list[list[RetrievedChunk]], rrf_k: int = 60) -> list[RetrievedChunk]:
    score: dict[str, float] = defaultdict(float)
    keep: dict[str, RetrievedChunk] = {}
    for lst in lists:
        for rank, c in enumerate(lst, start=1):
            score[_key(c)] += 1.0 / (rrf_k + rank)
            if _key(c) not in keep or c.similarity > keep[_key(c)].similarity:
                keep[_key(c)] = c
    return [keep[k] for k in sorted(score, key=lambda k: score[k], reverse=True)]


class Candidates:
    """Pool generators, so recall can be measured without any re-ranker."""

    def __init__(self, store: VectorStore, embedder: Embedder, bm25: BM25, depth: int = 40) -> None:
        self.store, self.embedder, self.bm25, self.depth = store, embedder, bm25, depth

    def embedding(self, q: str) -> list[RetrievedChunk]:
        emb = self.embedder.embed_query(q)
        return sorted(self.store.query(emb, k=self.depth, filters=None),
                      key=lambda c: c.similarity, reverse=True)

    def keyword(self, q: str) -> list[RetrievedChunk]:
        return self.bm25.top(q, self.depth)

    def hybrid(self, q: str) -> list[RetrievedChunk]:
        return fuse([self.embedding(q), self.keyword(q)])


class HybridRetriever:
    """Retriever.retrieve's signature, hybrid pool, production steps afterwards."""

    def __init__(self, cands: Candidates, cfg: QuarqConfig, reranker: RerankerLike,
                 pool_n: int) -> None:
        self.cands, self.cfg, self.reranker, self.pool_n = cands, cfg, reranker, pool_n

    def retrieve(self, query: str, k: int | None = None, doc_type: str | None = None,
                 min_similarity: float | None = None) -> list[RetrievedChunk]:
        k = k if k is not None else self.cfg.rag.top_k
        pool = self.cands.hybrid(query)
        head = self.reranker.rerank(query, pool[: self.pool_n])
        out = head + pool[self.pool_n:]
        if self.cfg.rag.date_aware:
            out = prefer_covering(out, exact_dates(query))
        return _best_chunk_per_page(out)[:k]
