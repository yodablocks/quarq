"""Exact dates in questions, and preferring documents whose period covers them.

The embedder and re-ranker don't weigh dates: for "the CAC 40 weight as of March 31,
2026" they can rank a June 2026 composition above the March 2026 factsheet. When a
question names an exact date, documents whose manifest period (period_start ..
period_end) contains it come first; the re-ranker's order is kept within each group.

The rule is deliberately narrow: vaguer wording ("in April 2025", "at the end of
2024") changes nothing, because the answer to such questions is often in a later
document (a review published after the events it describes).
"""

from __future__ import annotations

import re
from datetime import date

from quarq.rag.store import RetrievedChunk

_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6, "july": 7,
    "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
    "janvier": 1, "février": 2, "fevrier": 2, "mars": 3, "avril": 4, "mai": 5, "juin": 6,
    "juillet": 7, "août": 8, "aout": 8, "septembre": 9, "octobre": 10, "novembre": 11,
    "décembre": 12, "decembre": 12,
}
_MONTH = "|".join(sorted(_MONTHS, key=len, reverse=True))
_PATTERNS = [
    # March 31, 2026 / March 31 2026
    (re.compile(rf"\b({_MONTH})\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b", re.I), "mdy"),
    # 31 March 2026 / 31 décembre 2023 / 1er janvier 2025
    (re.compile(rf"\b(\d{{1,2}})(?:er|st|nd|rd|th)?\s+({_MONTH})\s+(\d{{4}})\b", re.I), "dmy"),
    # 2026-03-31
    (re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b"), "iso"),
]


def exact_dates(text: str) -> list[date]:
    """Return the full dates (day, month and year) named in a text, in order of appearance.

    Args:
        text: A question.

    Returns:
        Valid dates, without duplicates. Month-only or year-only mentions are ignored.
    """
    found: list[tuple[int, date]] = []
    for pattern, kind in _PATTERNS:
        for m in pattern.finditer(text):
            try:
                if kind == "mdy":
                    d = date(int(m[3]), _MONTHS[m[1].lower()], int(m[2]))
                elif kind == "dmy":
                    d = date(int(m[3]), _MONTHS[m[2].lower()], int(m[1]))
                else:
                    d = date(int(m[1]), int(m[2]), int(m[3]))
            except ValueError:  # e.g. February 30
                continue
            found.append((m.start(), d))
    out: list[date] = []
    for _, d in sorted(found, key=lambda x: x[0]):
        if d not in out:
            out.append(d)
    return out


def _covers(chunk: RetrievedChunk, dates: list[date]) -> bool:
    start, end = chunk.metadata.get("period_start"), chunk.metadata.get("period_end")
    if not start or not end:
        return False
    try:
        lo, hi = date.fromisoformat(str(start)), date.fromisoformat(str(end))
    except ValueError:
        return False
    return any(lo <= d <= hi for d in dates)


def prefer_covering(chunks: list[RetrievedChunk], dates: list[date]) -> list[RetrievedChunk]:
    """Move chunks whose document period covers one of the dates to the front.

    Order is kept within each group. If no chunk covers a date (for example a question
    about 2022 when the corpus starts in 2023), the order is unchanged.

    Args:
        chunks: Candidates, best first.
        dates: Exact dates from the question (see exact_dates).

    Returns:
        The reordered candidates.
    """
    if not dates:
        return chunks
    covering = [c for c in chunks if _covers(c, dates)]
    if not covering:
        return chunks
    return covering + [c for c in chunks if not _covers(c, dates)]
