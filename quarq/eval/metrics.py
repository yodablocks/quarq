"""Retrieval metrics over (source, page) references. Pure functions only."""

from __future__ import annotations

from collections.abc import Sequence

from quarq.eval.dataset import Ref
from quarq.exceptions import EvalError


def _check_k(k: int) -> None:
    if k < 1:
        raise EvalError(f"k must be >= 1, got {k}")


def precision_at_k(retrieved: Sequence[Ref], gold: set[Ref], k: int) -> float:
    """Share of the top-k retrieved items whose (source, page) is gold.

    Counts retrieved items, so two chunks from one gold page both count.
    Divides by the number actually returned (at most k), so a similarity
    floor that returns fewer than k chunks is not penalised for the gap.

    Args:
        retrieved: Retrieved references, best first.
        gold: Correct references for the question.
        k: Cut-off rank.

    Returns:
        Precision in [0, 1]; 0.0 when nothing was retrieved.

    Raises:
        EvalError: If k < 1.
    """
    _check_k(k)
    top = retrieved[:k]
    if not top:
        return 0.0
    return sum(1 for ref in top if ref in gold) / len(top)


def recall_at_k(retrieved: Sequence[Ref], gold: set[Ref], k: int) -> float:
    """Share of distinct gold references found in the top k.

    Args:
        retrieved: Retrieved references, best first.
        gold: Correct references for the question.
        k: Cut-off rank.

    Returns:
        Recall in [0, 1].

    Raises:
        EvalError: If k < 1 or gold is empty.
    """
    _check_k(k)
    if not gold:
        raise EvalError("gold set is empty")
    return len(set(retrieved[:k]) & gold) / len(gold)


def hit_at_k(retrieved: Sequence[Ref], gold: set[Ref], k: int) -> float:
    """Whether any gold reference appears in the top k.

    Args:
        retrieved: Retrieved references, best first.
        gold: Correct references for the question.
        k: Cut-off rank.

    Returns:
        1.0 on a hit, else 0.0 (float so it averages cleanly).

    Raises:
        EvalError: If k < 1.
    """
    _check_k(k)
    return 1.0 if any(ref in gold for ref in retrieved[:k]) else 0.0


def reciprocal_rank(retrieved: Sequence[Ref], gold: set[Ref]) -> float:
    """1 / rank of the first gold reference, or 0.0 if none is retrieved.

    Args:
        retrieved: Retrieved references, best first.
        gold: Correct references for the question.

    Returns:
        Reciprocal rank in [0, 1].
    """
    for rank, ref in enumerate(retrieved, start=1):
        if ref in gold:
            return 1.0 / rank
    return 0.0


def mean(values: Sequence[float]) -> float:
    """Arithmetic mean, 0.0 for an empty sequence.

    Args:
        values: Numbers to average.

    Returns:
        The mean.
    """
    return sum(values) / len(values) if values else 0.0
