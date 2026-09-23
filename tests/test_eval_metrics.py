"""Tests for quarq.eval.metrics, checked against hand-computed values."""

from __future__ import annotations

import pytest

from quarq.eval.metrics import (
    hit_at_k,
    mean,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from quarq.exceptions import EvalError

A = ("a.pdf", 1)
B = ("b.pdf", 2)
C = ("c.pdf", 3)
D = ("d.pdf", 4)


def test_spec_example_single_gold() -> None:
    retrieved = [B, A, C]
    gold = {A}
    assert precision_at_k(retrieved, gold, 1) == 0.0
    assert precision_at_k(retrieved, gold, 3) == pytest.approx(1 / 3)
    assert recall_at_k(retrieved, gold, 1) == 0.0
    assert recall_at_k(retrieved, gold, 3) == 1.0
    assert hit_at_k(retrieved, gold, 1) == 0.0
    assert hit_at_k(retrieved, gold, 3) == 1.0
    assert reciprocal_rank(retrieved, gold) == 0.5


def test_precision_divides_by_returned_count_when_fewer_than_k() -> None:
    # Only 2 chunks survived the similarity floor; one is relevant.
    assert precision_at_k([A, B], {A}, 5) == 0.5


def test_multi_ref_gold_recall_counts_distinct_pages() -> None:
    retrieved = [A, B, C]
    gold = {A, C, D}
    assert recall_at_k(retrieved, gold, 3) == pytest.approx(2 / 3)
    assert recall_at_k(retrieved, gold, 1) == pytest.approx(1 / 3)


def test_duplicate_page_counts_per_item_for_precision_once_for_recall() -> None:
    # Two chunks from the same gold page.
    retrieved = [A, A, B]
    gold = {A, C}
    assert precision_at_k(retrieved, gold, 3) == pytest.approx(2 / 3)
    assert recall_at_k(retrieved, gold, 3) == 0.5


def test_empty_retrieval_is_a_miss() -> None:
    assert precision_at_k([], {A}, 5) == 0.0
    assert recall_at_k([], {A}, 5) == 0.0
    assert hit_at_k([], {A}, 5) == 0.0
    assert reciprocal_rank([], {A}) == 0.0


def test_invalid_k_raises() -> None:
    with pytest.raises(EvalError, match="k must be >= 1"):
        precision_at_k([A], {A}, 0)


def test_empty_gold_raises_for_recall() -> None:
    with pytest.raises(EvalError, match="gold set is empty"):
        recall_at_k([A], set(), 1)


def test_mean() -> None:
    assert mean([1.0, 0.0, 0.5]) == 0.5
    assert mean([]) == 0.0
