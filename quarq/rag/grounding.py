"""Check that the figures in an answer appear in what the model was shown.

The research agent is told to answer only from the retrieved passages, but nothing
enforces it. This is a deterministic first check: every figure in the answer (counts,
amounts, percentages, years) must appear somewhere in the prompt the model received.
It does not check wording or reasoning, only figures.

Number formats differ between the sources and the answer ("134 803", "134,803" and
"134803"; "7,6 %" and "7.6%"), so each figure is read under both the English
(',' thousands, '.' decimal) and the French ('.' or space thousands, ',' decimal)
conventions, and counts as supported if any reading matches.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation

_SPACES = "    "
_NUMBER = re.compile(
    r"(?<![\w.,])(?:"
    r"\d{1,3}(?:[    ]\d{3})+(?:[.,]\d+)?"  # 8 813 / 134 803
    r"|\d{1,3}(?:,\d{3})+(?:\.\d+)?"  # 134,803 / 590,000.5
    r"|\d{1,3}(?:\.\d{3})+(?:,\d+)?"  # 1.234.567,8
    r"|\d+(?:[.,]\d+)?"  # 2024 / 4.98 / 7,6
    r")(?!\d)"
)
_PERCENT = re.compile(r"^[   ]?%")
_MAGNITUDE = re.compile(
    r"^\s*(thousand|million|billion|trillion|bn|mn|mille|milliers?|millions?|milliards?)\b", re.I
)
_CURRENCY = re.compile(r"(€|\$|£|\bUSD|\bEUR|\bGBP)\s*$")


def _readings(raw: str) -> set[str]:
    """Return the canonical decimal values a written number can mean."""
    compact = raw
    for ch in _SPACES:
        compact = compact.replace(ch, "")
    out: set[str] = set()
    for thousands, decimal in ((",", "."), (".", ",")):
        head, sep, tail = compact.rpartition(decimal)
        integer = head if sep else compact
        if thousands in (tail if sep else ""):
            continue
        groups = integer.split(thousands)
        if len(groups) > 1 and not all(len(g) == 3 for g in groups[1:]):
            continue
        text = "".join(groups) + ("." + tail if sep else "")
        try:
            value = Decimal(text).normalize()
        except InvalidOperation:
            continue
        out.add(format(value, "f"))
    return out


@dataclass
class GroundingReport:
    """Figures found in an answer, and those missing from the sources.

    Attributes:
        checked: Distinct figures in the answer that were checked.
        unsupported: Figures (as written in the answer) found nowhere in the sources.
    """

    checked: int = 0
    unsupported: list[str] = field(default_factory=list)

    @property
    def grounded(self) -> bool:
        """True if every checked figure appears in the sources."""
        return not self.unsupported


def _answer_figures(text: str) -> list[tuple[str, set[str]]]:
    """Figures worth checking in an answer: skip citation markers and bare small integers."""
    figures: list[tuple[str, set[str]]] = []
    for m in _NUMBER.finditer(text):
        raw = m.group(0)
        before, after = text[: m.start()], text[m.end():]
        if before.endswith("[") and after.startswith("]"):
            continue  # a citation marker like [2]
        pct = _PERCENT.match(after)
        significant = (
            len(raw) > 1 or pct or _MAGNITUDE.match(after) or _CURRENCY.search(before[-6:])
        )
        if not significant:
            continue  # "3 reasons", "point 4": too common to check
        figures.append((raw + (pct.group(0) if pct else ""), _readings(raw)))
    return figures


def check_figures(answer: str, sources: str) -> GroundingReport:
    """Check that every figure in the answer appears in the sources.

    Args:
        answer: The model's answer.
        sources: Everything the model was shown (the full prompt, including the question).

    Returns:
        The GroundingReport.
    """
    known: set[str] = set()
    for m in _NUMBER.finditer(sources):
        known |= _readings(m.group(0))
    report = GroundingReport()
    seen: set[frozenset[str]] = set()
    for raw, readings in _answer_figures(answer):
        key = frozenset(readings)
        if key in seen:
            continue
        seen.add(key)
        report.checked += 1
        if not readings & known:
            report.unsupported.append(raw)
    return report
