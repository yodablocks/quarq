"""Tests for exact-date parsing and date-aware ordering of retrieved chunks."""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

import pytest

from quarq.rag.store import RetrievedChunk


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("What was TotalEnergies' weight in the CAC 40 as of March 31, 2026?", [date(2026, 3, 31)]),
        ("What was the weight of LVMH in the CAC 40 on June 30, 2026?", [date(2026, 6, 30)]),
        ("Weight on 30 June 2026 and on 2026-03-31?", [date(2026, 6, 30), date(2026, 3, 31)]),
        ("Combien de salariés la Banque de France comptait-elle au 31 décembre 2023 ?",
         [date(2023, 12, 31)]),
        ("au 1er janvier 2025", [date(2025, 1, 1)]),
        ("en février 2024, le 29 février 2024", [date(2024, 2, 29)]),
    ],
)
def test_exact_dates_are_found(text: str, expected: list[date]) -> None:
    from quarq.rag.dates import exact_dates

    assert exact_dates(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "How many market intermediaries did the AMF count at the end of 2024?",
        "Combien de dossiers de surendettement ont été déposés en 2023 ?",
        "What happened after the tariff announcements in April 2025?",
        "on February 30, 2026",  # not a real date
    ],
)
def test_vague_or_invalid_dates_are_ignored(text: str) -> None:
    from quarq.rag.dates import exact_dates

    assert exact_dates(text) == []


def _chunk(name: str, start: str | None, end: str | None, page: int = 1) -> RetrievedChunk:
    meta = {"source": name, "page": page}
    if start:
        meta |= {"period_start": start, "period_end": end}
    return RetrievedChunk(content=name, metadata=meta, similarity=0.9, source=name, page=page)


def test_documents_covering_the_date_come_first_keeping_their_order() -> None:
    from quarq.rag.dates import prefer_covering

    june = _chunk("composition_june", "2026-06-30", "2026-06-30")
    march_a = _chunk("factsheet_march", "2026-03-31", "2026-03-31", page=3)
    h1 = _chunk("review_h1", "2026-01-01", "2026-06-30")
    march_b = _chunk("factsheet_march", "2026-03-31", "2026-03-31", page=1)

    out = prefer_covering([june, march_a, h1, march_b], [date(2026, 3, 31)])

    assert [(c.source, c.page) for c in out] == [
        ("factsheet_march", 3), ("review_h1", 1), ("factsheet_march", 1), ("composition_june", 1)]


def test_no_change_when_nothing_covers_the_date() -> None:
    from quarq.rag.dates import prefer_covering

    chunks = [
        _chunk("key_figures_2023", "2023-01-01", "2023-12-31"),
        _chunk("no_period", None, None),
    ]
    assert prefer_covering(chunks, [date(2022, 12, 31)]) == chunks


def test_no_change_without_a_date() -> None:
    from quarq.rag.dates import prefer_covering

    chunks = [_chunk("b", "2025-01-01", "2025-12-31"), _chunk("a", "2024-01-01", "2024-12-31")]
    assert prefer_covering(chunks, []) == chunks


def test_retriever_applies_date_preference_after_reranking() -> None:
    from quarq.config import QuarqConfig
    from quarq.rag.retriever import Retriever

    cfg = QuarqConfig()
    june = _chunk("composition_june", "2026-06-30", "2026-06-30")
    march = _chunk("factsheet_march", "2026-03-31", "2026-03-31", page=3)
    store = MagicMock()
    store.query.return_value = [march, june]
    embedder = MagicMock()
    embedder.embed_query.return_value = [0.1]
    reranker = MagicMock()
    reranker.rerank.side_effect = lambda q, chunks: [chunks[1], chunks[0]]  # June first

    retriever = Retriever(store, embedder, cfg, reranker=reranker)
    out = retriever.retrieve("TotalEnergies weight as of March 31, 2026?", k=2)
    assert [c.source for c in out] == ["factsheet_march", "composition_june"]

    cfg.rag.date_aware = False
    out = retriever.retrieve("TotalEnergies weight as of March 31, 2026?", k=2)
    assert [c.source for c in out] == ["composition_june", "factsheet_march"]
