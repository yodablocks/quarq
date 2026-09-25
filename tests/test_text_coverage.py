"""Tests for detecting pages without a text layer (blank, image-only, thin)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch


def _page(n: int, text: str, images: int) -> MagicMock:
    page = MagicMock()
    page.page_number = n
    page.extract_text.return_value = text
    page.images = [{}] * images
    return page


def _pdf(pages: list[MagicMock]) -> MagicMock:
    pdf = MagicMock()
    pdf.__enter__ = MagicMock(return_value=pdf)
    pdf.__exit__ = MagicMock(return_value=False)
    pdf.pages = pages
    pdf.metadata = {}
    return pdf


BODY = "Financial stability text. " * 40


def test_classify_page_kinds() -> None:
    from quarq.rag.loader import classify_page

    assert classify_page(_page(1, BODY, 0)) == "text"
    assert classify_page(_page(1, "", 0)) == "blank"
    assert classify_page(_page(1, "   ", 2)) == "image_only"
    assert classify_page(_page(1, "Chart 1.2", 1)) == "thin"
    assert classify_page(_page(1, "Short heading", 0)) == "text"  # little text, no image


def test_load_pdf_fills_coverage_in_the_same_pass(tmp_path: Path) -> None:
    from quarq.rag.loader import TextCoverage, load_pdf

    pdf_path = tmp_path / "report.pdf"
    pdf_path.write_bytes(b"%PDF")
    pages = [_page(1, BODY, 0), _page(2, "", 0), _page(3, "", 3), _page(4, "Fig", 1)]
    coverage = TextCoverage(source="")

    with patch("pdfplumber.open", return_value=_pdf(pages)):
        docs = load_pdf(pdf_path, coverage=coverage)

    assert coverage.source == "report.pdf"
    assert coverage.pages == 4
    assert coverage.blank == [2]
    assert coverage.image_only == [3]
    assert coverage.thin == [4]
    assert {d.metadata["page"] for d in docs} == {1, 4}  # only pages with text are chunked
    assert not coverage.likely_scanned


def test_mostly_image_only_document_is_flagged_as_scanned(tmp_path: Path) -> None:
    from quarq.rag.loader import text_coverage

    pdf_path = tmp_path / "scan.pdf"
    pdf_path.write_bytes(b"%PDF")
    pages = [_page(i, "", 1) for i in range(1, 9)] + [_page(9, BODY, 0)]

    with patch("pdfplumber.open", return_value=_pdf(pages)):
        coverage = text_coverage(pdf_path)

    assert coverage.image_only == list(range(1, 9))
    assert coverage.likely_scanned


def test_load_folder_collects_one_coverage_per_file(tmp_path: Path) -> None:
    from quarq.rag.loader import load_folder

    for name in ("a.pdf", "b.pdf"):
        (tmp_path / name).write_bytes(b"%PDF")
    coverages: list = []

    with patch("pdfplumber.open", side_effect=lambda p: _pdf([_page(1, BODY, 0), _page(2, "", 1)])):
        load_folder(tmp_path, coverages=coverages)

    assert sorted(c.source for c in coverages) == ["a.pdf", "b.pdf"]
    assert all(c.image_only == [2] for c in coverages)


def test_rag_add_warning_names_scanned_files(capsys) -> None:
    from quarq import cli
    from quarq.rag.loader import TextCoverage

    scanned = TextCoverage(source="scan.pdf", pages=4, image_only=[1, 2, 3, 4])
    partial = TextCoverage(source="report.pdf", pages=10, image_only=[7])
    cli._warn_on_missing_text([scanned, partial])

    out = " ".join(capsys.readouterr().out.split())
    assert "Likely scanned" in out and "scan.pdf" in out
    assert "report.pdf" not in out.split("Likely scanned")[1].split("page(s)")[0]
    assert "5 page(s) have images but no text" in out


def test_rag_add_warning_is_silent_when_all_pages_have_text(capsys) -> None:
    from quarq import cli
    from quarq.rag.loader import TextCoverage

    cli._warn_on_missing_text([TextCoverage(source="ok.pdf", pages=3)])

    assert capsys.readouterr().out == ""
