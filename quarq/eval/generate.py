"""Offline gold-set drafting: an LLM proposes questions, a human accepts them.

Nothing here runs during `quarq eval`. Drafts carry provenance
"synthetic-draft", which load_gold refuses until a human reviews each row.
"""

from __future__ import annotations

import random
import re
from collections import defaultdict
from collections.abc import Sequence
from typing import Protocol

from quarq.constants import EVAL_MIN_CHUNK_CHARS
from quarq.eval.dataset import DRAFT_PROVENANCE, GoldItem, GoldRef
from quarq.rag.store import RetrievedChunk

QUESTION_SYSTEM = (
    "You write evaluation questions for a retrieval system over central-bank "
    "financial-stability reports and market-regulator publications."
)

_PREFIX = re.compile(r"^\s*(?:\d+[.)]\s*|q(?:uestion)?\s*[:.-]\s*)", re.IGNORECASE)


class LLMLike(Protocol):
    """The subset of quarq.llm.BaseLLM the generator depends on."""

    def generate(self, prompt: str, system: str = "") -> str: ...


def build_question_prompt(passage: str) -> str:
    """Build the prompt asking for one question the passage answers.

    Args:
        passage: Chunk text.

    Returns:
        Prompt text.
    """
    return (
        "Write ONE specific question that the passage below answers. The question must be "
        "answerable from this passage alone, must name its concrete subject (institution, "
        "metric, market, or year), and must not copy a sentence from the passage. Reply with "
        "the question only, on one line.\n\nPassage:\n" + passage
    )


def clean_question(raw: str) -> str | None:
    """Extract a single question from raw LLM output.

    Args:
        raw: Model reply.

    Returns:
        The question, or None if the reply holds no question.
    """
    for line in raw.splitlines():
        text = _PREFIX.sub("", line.strip()).strip().strip("\"'").strip()
        if text:
            return text if text.endswith("?") else None
    return None


def sample_chunks(
    chunks: Sequence[RetrievedChunk],
    per_doc_type: int,
    seed: int,
    min_chars: int = EVAL_MIN_CHUNK_CHARS,
) -> list[RetrievedChunk]:
    """Pick up to per_doc_type eligible chunks from each doc_type, reproducibly.

    Args:
        chunks: Every stored chunk.
        per_doc_type: Maximum chunks to draw per doc_type.
        seed: Random seed; the same seed and corpus give the same sample.
        min_chars: Skip chunks shorter than this (headers, tables of contents).

    Returns:
        Sampled chunks grouped by doc_type in sorted doc_type order.
    """
    groups: dict[str, list[RetrievedChunk]] = defaultdict(list)
    for chunk in chunks:
        if len(chunk.content.strip()) >= min_chars:
            groups[str(chunk.metadata.get("doc_type", "unknown"))].append(chunk)

    rng = random.Random(seed)
    sampled: list[RetrievedChunk] = []
    for doc_type in sorted(groups):
        pool = sorted(groups[doc_type], key=lambda c: str(c.metadata.get("chunk_id", "")))
        sampled.extend(rng.sample(pool, min(per_doc_type, len(pool))))
    return sampled


def draft_items(chunks: Sequence[RetrievedChunk], llm: LLMLike) -> tuple[list[GoldItem], int]:
    """Ask the LLM for one question per chunk and label it with the chunk's page.

    Args:
        chunks: Chunks to draft from.
        llm: Backend with a generate(prompt, system) method.

    Returns:
        (draft items, number of chunks skipped because no question came back).

    Raises:
        RAGError: Propagated from the LLM backend.
    """
    items: list[GoldItem] = []
    skipped = 0
    for chunk in chunks:
        question = clean_question(
            llm.generate(build_question_prompt(chunk.content), system=QUESTION_SYSTEM)
        )
        if question is None:
            skipped += 1
            continue
        chunk_id = str(chunk.metadata.get("chunk_id", ""))[:12]
        doc_type = chunk.metadata.get("doc_type")
        items.append(
            GoldItem(
                id=f"draft{len(items) + 1:03d}",
                question=question,
                gold=[GoldRef(source=chunk.source, page=int(chunk.page))],
                doc_type=str(doc_type) if doc_type else None,
                note=(
                    f"drafted from chunk {chunk_id}. Check the answer is on this page; "
                    "add any other pages that also answer it."
                ),
                provenance=DRAFT_PROVENANCE,
            )
        )
    return items, skipped
