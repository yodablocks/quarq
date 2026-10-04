"""Re-rank retrieved chunks with one Jev Noul per (question, passage) pair."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

from experiments.jev_rerank.client import JevClient
from quarq.rag.store import RetrievedChunk

QUESTION = {
    "relevant": {
        "type": "noul",
        "instructions": (
            "Does `passage` itself state the specific information that answers `question`? "
            "Answer yes only if the answer is written in the passage. A passage that is about "
            "the same topic, edition or period but does not contain the answer is a no."
        ),
        "criteria": {
            "true": "The passage states the answer to the question.",
            "false": "The passage does not state the answer, even if it covers the same topic.",
        },
    }
}


class JevReranker:
    """Same interface as quarq.rag.reranker.Reranker."""

    def __init__(self, client: JevClient, workers: int = 12) -> None:
        self.client = client
        self.workers = workers
        self.ties_at_top = 0
        self.calls_wall_ms: list[float] = []

    def _score(self, query: str, chunk: RetrievedChunk) -> float:
        out = self.client.ask({"question": query, "passage": chunk.content}, QUESTION)
        return float(out["answers"]["relevant"]["noul"])

    def rerank(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if not chunks:
            return []
        import time

        started = time.time()
        with ThreadPoolExecutor(self.workers) as pool:
            scores = list(pool.map(lambda c: self._score(query, c), chunks))
        self.calls_wall_ms.append((time.time() - started) * 1000)
        if sum(1 for s in scores if s == max(scores)) > 1:
            self.ties_at_top += 1
        scored = [replace(c, rerank_score=s) for c, s in zip(chunks, scores, strict=True)]
        return sorted(scored, key=lambda c: c.rerank_score or 0.0, reverse=True)  # stable
