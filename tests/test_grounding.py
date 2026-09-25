"""Tests for checking that figures in an answer appear in what the model was shown."""

from __future__ import annotations

import pytest

SOURCE = (
    "[1] Source: bdf_rapport_annuel_2024.pdf, Page: 155\n"
    "Au 31 décembre 2024, la Banque de France comptait 8 813 salariés (en équivalent "
    "temps plein). Le nombre de dossiers déposés est de 134 803, soit une hausse de 10,8 % "
    "par rapport à 2023.\n"
    "[2] Source: amf_ra_2024_veng.pdf, Page: 74\n"
    "The Enforcement Committee imposed fines totalling €590,000. BNP PARIBAS weight 4.98%. "
    "Crypto market capitalisation breached the USD 4 trillion mark. Unemployment 7,6 %."
)


@pytest.mark.parametrize(
    "answer",
    [
        "La Banque de France comptait 8 813 salariés au 31 décembre 2024.",
        "134,803 files were filed in 2024, up 10.8% on 2023.",
        "Les dossiers déposés : 134803.",
        "The fines totalled €590,000 [2].",
        "BNP Paribas weighed 4.98 % (Page 74).",
        "The unemployment rate was 7.6%.",
        "It passed USD 4 trillion.",
        "Selon le document [1], page 155, l'effectif était de 8813.",
    ],
)
def test_figures_present_in_the_sources_are_supported(answer: str) -> None:
    from quarq.rag.grounding import check_figures

    report = check_figures(answer, SOURCE)

    assert report.unsupported == [], report.unsupported
    assert report.checked >= 1


@pytest.mark.parametrize(
    ("answer", "bad"),
    [
        ("La Banque de France comptait 8 959 salariés.", "8 959"),
        ("134 802 dossiers ont été déposés.", "134 802"),
        ("The fines totalled €950,000.", "950,000"),
        ("BNP Paribas weighed 5.86%.", "5.86%"),
        ("Unemployment was 7.4%.", "7.4%"),
        ("In 2022 the figure was lower.", "2022"),
    ],
)
def test_figures_missing_from_the_sources_are_flagged(answer: str, bad: str) -> None:
    from quarq.rag.grounding import check_figures

    report = check_figures(answer, SOURCE)

    assert report.unsupported == [bad]
    assert not report.grounded


def test_small_bare_integers_and_citation_markers_are_ignored() -> None:
    from quarq.rag.grounding import check_figures

    report = check_figures("There are 3 reasons, see [1] and [2]; point 4 matters.", SOURCE)

    assert report.checked == 0
    assert report.grounded


def test_answer_without_figures_is_reported_as_nothing_checked() -> None:
    from quarq.rag.grounding import check_figures

    report = check_figures("The AMF asked AFEP and MEDEF to look into it.", SOURCE)

    assert report.checked == 0 and report.unsupported == []


def test_each_unsupported_figure_is_listed_once() -> None:
    from quarq.rag.grounding import check_figures

    report = check_figures("9 999 then 9 999 again, and 8 813.", SOURCE)

    assert report.unsupported == ["9 999"]
    assert report.checked == 2


def test_answer_attaches_the_grounding_check(monkeypatch: pytest.MonkeyPatch) -> None:
    from quarq.config import QuarqConfig
    from quarq.rag import generator
    from quarq.rag.store import RetrievedChunk

    class _LLM:
        name, model = "fake", "fake-model"

        def generate(self, prompt: str, system: str = "") -> str:
            return "There were 8 813 employees, and 12 345 contractors."

    monkeypatch.setattr(generator, "get_llm", lambda cfg, agent: _LLM())
    chunk = RetrievedChunk(content="La Banque comptait 8 813 salariés.", metadata={},
                           similarity=0.9, source="r.pdf", page=155)

    result = generator.answer("How many employees?", [chunk], QuarqConfig())

    assert result.figures_checked == 2
    assert result.unsupported_figures == ["12 345"]
