"""Prompt construction, LLM routing, and answer generation for RAG queries."""

from __future__ import annotations

import time

from quarq.config import QuarqConfig
from quarq.llm.base import GenerationResult
from quarq.llm.lmstudio import get_llm
from quarq.rag.store import RetrievedChunk

_SYSTEM_PROMPT = (
    "You are a senior portfolio analyst at a French institutional investor. "
    "Answer only from the provided context. If the context does not contain "
    "enough information to answer, say so explicitly. Never hallucinate citations."
)


def _build_prompt(
    query: str,
    chunks: list[RetrievedChunk],
    portfolio_context: str = "",
) -> str:
    """Assemble the full prompt from retrieved chunks and optional portfolio context.

    Args:
        query: The user's question.
        chunks: Retrieved document chunks (already filtered by similarity).
        portfolio_context: Optional portfolio metrics or narrative context.

    Returns:
        Formatted prompt string ready to send to the LLM.
    """
    parts: list[str] = []

    if portfolio_context:
        parts.append("PORTFOLIO CONTEXT:")
        parts.append(portfolio_context)
        parts.append("")

    parts.append("RETRIEVED DOCUMENTS:")
    for i, chunk in enumerate(chunks[:3], start=1):
        parts.append(f"[{i}] Source: {chunk.source}, Page: {chunk.page}")
        parts.append(chunk.content[:500])
        parts.append("")

    parts.append(query)
    return "\n".join(parts)


def format_citations(chunks: list[RetrievedChunk]) -> str:
    """Format retrieved chunks as a citations string.

    Args:
        chunks: Retrieved chunks to cite.

    Returns:
        Multi-line string listing source, page, and similarity for each chunk.
    """
    lines = [
        f"Source: {c.source}, Page {c.page} (similarity: {c.similarity:.2f})"
        for c in chunks
    ]
    return "\n".join(lines)


def answer(
    query: str,
    chunks: list[RetrievedChunk],
    cfg: QuarqConfig,
    portfolio_context: str = "",
) -> GenerationResult:
    """Generate a grounded answer using the research LLM agent.

    Args:
        query: The user's natural language question.
        chunks: Retrieved document chunks from the retriever.
        cfg: Loaded QuarqConfig (used for LLM routing).
        portfolio_context: Optional portfolio summary to include in the prompt.

    Returns:
        GenerationResult with the answer text, model, backend, and latency.

    Raises:
        RAGError: If no LLM backend is available or generation fails.
    """
    llm = get_llm(cfg, agent="research")
    prompt = _build_prompt(query, chunks, portfolio_context)

    start = time.monotonic()
    response_text = llm.generate(prompt, system=_SYSTEM_PROMPT)
    latency_ms = int((time.monotonic() - start) * 1000)

    return GenerationResult(
        answer=response_text,
        model=llm.model,
        backend=llm.name,
        latency_ms=latency_ms,
    )
