"""Tests for the corpus manifest: parsing, loader overrides, and applying to a real store."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from quarq.exceptions import RAGError

VALID = """
[[document]]
source = "bdf_rapport_annuel_2024.pdf"
doc_type = "bdf_fsr"
published = 2025-03-17
period_start = 2024-01-01
period_end = 2024-12-31

[[document]]
source = "factsheet_cac40_index.pdf"
period_start = 2026-03-31
period_end = 2026-03-31
"""


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "quarq_manifest.toml"
    path.write_text(text, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Parsing and validation
# ---------------------------------------------------------------------------


def test_load_manifest_parses_entries(tmp_path: Path) -> None:
    from quarq.rag.manifest import load_manifest

    manifest = load_manifest(_write(tmp_path, VALID))

    assert set(manifest) == {"bdf_rapport_annuel_2024.pdf", "factsheet_cac40_index.pdf"}
    entry = manifest["bdf_rapport_annuel_2024.pdf"]
    assert entry.published == date(2025, 3, 17)
    assert entry.period_start == date(2024, 1, 1)
    assert manifest["factsheet_cac40_index.pdf"].doc_type is None


def test_entry_metadata_uses_iso_strings_and_only_set_fields(tmp_path: Path) -> None:
    from quarq.rag.manifest import load_manifest

    manifest = load_manifest(_write(tmp_path, VALID))

    assert manifest["bdf_rapport_annuel_2024.pdf"].metadata() == {
        "doc_type": "bdf_fsr",
        "date": "2025-03-17",
        "period_start": "2024-01-01",
        "period_end": "2024-12-31",
    }
    # doc_type and published not set: they must not overwrite the loader's values
    assert manifest["factsheet_cac40_index.pdf"].metadata() == {
        "period_start": "2026-03-31",
        "period_end": "2026-03-31",
    }


@pytest.mark.parametrize(
    ("snippet", "expected"),
    [
        ('colour = "blue"', "colour"),
        ('doc_type = "ecb_fsrr"', "doc_type"),
        ("period_start = 2025-01-01\nperiod_end = 2024-01-01", "period_start"),
    ],
)
def test_invalid_entry_names_source_and_field(
    tmp_path: Path, snippet: str, expected: str
) -> None:
    from quarq.rag.manifest import load_manifest

    base = 'source = "x.pdf"\n'
    if "period_start" not in snippet:
        base += "period_start = 2024-01-01\nperiod_end = 2024-12-31\n"
    text = "[[document]]\n" + base + snippet + "\n"

    with pytest.raises(RAGError) as excinfo:
        load_manifest(_write(tmp_path, text))
    message = str(excinfo.value)
    assert "x.pdf" in message
    assert expected in message


def test_missing_period_is_rejected(tmp_path: Path) -> None:
    from quarq.rag.manifest import load_manifest

    with pytest.raises(RAGError, match="period_end"):
        load_manifest(
            _write(tmp_path, '[[document]]\nsource = "x.pdf"\nperiod_start = 2024-01-01\n')
        )


def test_duplicate_source_is_rejected(tmp_path: Path) -> None:
    from quarq.rag.manifest import load_manifest

    entry = '[[document]]\nsource = "x.pdf"\nperiod_start = 2024-01-01\nperiod_end = 2024-12-31\n'
    with pytest.raises(RAGError, match="duplicate source 'x.pdf'"):
        load_manifest(_write(tmp_path, entry + entry))


def test_unknown_top_level_key_is_rejected(tmp_path: Path) -> None:
    from quarq.rag.manifest import load_manifest

    with pytest.raises(RAGError, match="documents"):
        load_manifest(_write(tmp_path, '[[documents]]\nsource = "x.pdf"\n'))


def test_find_manifest_for_file_and_folder(tmp_path: Path) -> None:
    from quarq.rag.manifest import find_manifest

    path = _write(tmp_path, VALID)
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"%PDF")

    assert find_manifest(tmp_path) == path
    assert find_manifest(pdf) == path
    assert find_manifest(tmp_path / "missing_dir") is None


# ---------------------------------------------------------------------------
# Loader applies the manifest
# ---------------------------------------------------------------------------


def _mock_pdf() -> MagicMock:
    page = MagicMock()
    page.extract_text.return_value = "Banque de France annual report text. " * 30
    page.page_number = 1
    pdf = MagicMock()
    pdf.__enter__ = MagicMock(return_value=pdf)
    pdf.__exit__ = MagicMock(return_value=False)
    pdf.pages = [page]
    pdf.metadata = {"CreationDate": "D:20250317120000"}
    return pdf


def test_load_pdf_applies_manifest_entry(tmp_path: Path) -> None:
    from quarq.rag.loader import load_pdf
    from quarq.rag.manifest import load_manifest

    manifest = load_manifest(_write(tmp_path, VALID.replace("bdf_fsr", "macro")))
    pdf_path = tmp_path / "bdf_rapport_annuel_2024.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")

    with patch("pdfplumber.open", return_value=_mock_pdf()):
        docs = load_pdf(pdf_path, manifest=manifest)

    meta = docs[0].metadata
    assert meta["doc_type"] == "macro"  # manifest override beats the filename rule
    assert meta["date"] == "2025-03-17"
    assert meta["period_start"] == "2024-01-01"
    assert meta["period_end"] == "2024-12-31"
    assert {"source", "page", "chunk_id"} <= set(meta)


def test_load_pdf_without_manifest_entry_keeps_inference(tmp_path: Path) -> None:
    from quarq.rag.loader import load_pdf

    pdf_path = tmp_path / "bdf_rapport_annuel_2024.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")

    with patch("pdfplumber.open", return_value=_mock_pdf()):
        docs = load_pdf(pdf_path, manifest={})

    meta = docs[0].metadata
    assert meta["doc_type"] == "bdf_fsr"
    assert meta["date"] == "2025-03-17"
    assert "period_start" not in meta


# ---------------------------------------------------------------------------
# Applying to an existing index (real ChromaDB in tmp_path)
# ---------------------------------------------------------------------------


def _real_store(tmp_path: Path):
    from quarq.config import QuarqConfig
    from quarq.rag.loader import Document
    from quarq.rag.store import VectorStore

    cfg = QuarqConfig()
    cfg.rag.chroma_path = str(tmp_path / "chroma")
    store = VectorStore(cfg)
    docs = [
        Document(
            content=f"chunk {i}",
            metadata={"source": src, "doc_type": "bdf_fsr", "date": "2025-03-17",
                      "page": page, "chunk_id": f"id{i}"},
        )
        for i, (src, page) in enumerate(
            [("bdf_rapport_annuel_2024.pdf", 1), ("bdf_rapport_annuel_2024.pdf", 2),
             ("other.pdf", 1)]
        )
    ]
    store.upsert(docs, [[0.1 * (i + 1), 0.2, 0.3] for i in range(len(docs))])
    return store


def _metas(store) -> dict[str, dict]:
    got = store._collection.get(include=["metadatas"])
    return dict(zip(got["ids"], got["metadatas"], strict=True))


def test_apply_manifest_updates_only_listed_sources(tmp_path: Path) -> None:
    from quarq.rag.manifest import apply_manifest, load_manifest

    store = _real_store(tmp_path)
    before = _metas(store)
    manifest = load_manifest(_write(tmp_path, VALID))

    report = apply_manifest(store, manifest)

    after = _metas(store)
    for cid in ("id0", "id1"):
        assert after[cid]["period_start"] == "2024-01-01"
        assert after[cid]["period_end"] == "2024-12-31"
        # the original five keys survive the update
        for key in ("source", "doc_type", "date", "page", "chunk_id"):
            assert key in after[cid]
        assert after[cid]["page"] == before[cid]["page"]
    assert after["id2"] == before["id2"]  # a source missing from the manifest is untouched
    assert report.updated == {"bdf_rapport_annuel_2024.pdf": 2}
    assert report.not_indexed == ["factsheet_cac40_index.pdf"]
    assert report.not_in_manifest == ["other.pdf"]


def test_apply_manifest_is_idempotent(tmp_path: Path) -> None:
    from quarq.rag.manifest import apply_manifest, load_manifest

    store = _real_store(tmp_path)
    manifest = load_manifest(_write(tmp_path, VALID))

    apply_manifest(store, manifest)
    once = _metas(store)
    apply_manifest(store, manifest)

    assert _metas(store) == once
