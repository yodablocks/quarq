"""PDF text extraction and chunking with required metadata attachment."""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pdfplumber

from quarq.constants import SCANNED_IMAGE_ONLY_SHARE, TEXT_THIN_PAGE_CHARS

if TYPE_CHECKING:
    from quarq.rag.manifest import Manifest

logger = logging.getLogger(__name__)

# Filename substring -> doc_type mapping (checked in order)
_DOC_TYPE_RULES: list[tuple[str, str]] = [
    ("ecb", "ecb_fsr"),
    ("amf", "amf_sfdr"),
    ("sfdr", "amf_sfdr"),
    ("prospectus", "prospectus"),
    ("factsheet", "factsheet"),
    ("cac40", "factsheet"),
    ("banque", "bdf_fsr"),
    ("bdf", "bdf_fsr"),
]


@dataclass
class Document:
    """A single text chunk extracted from a PDF.

    Attributes:
        content: The chunk text.
        metadata: Dict with keys: source, doc_type, date, page, chunk_id.
    """

    content: str
    metadata: dict[str, str | int] = field(default_factory=dict)


@dataclass
class TextCoverage:
    """How much of a PDF has a text layer quarq can index.

    Attributes:
        source: PDF filename.
        pages: Total pages.
        blank: Pages with no text and no images (usually intentional blank pages).
        image_only: Pages with images but no text: scans, full-page photos or
            infographics. Their content isn't searchable without OCR.
        thin: Pages with images and very little text, such as a chart with a caption.
    """

    source: str
    pages: int = 0
    blank: list[int] = field(default_factory=list)
    image_only: list[int] = field(default_factory=list)
    thin: list[int] = field(default_factory=list)

    @property
    def likely_scanned(self) -> bool:
        """True if most pages are image-only, so the document looks scanned."""
        return self.pages > 0 and len(self.image_only) / self.pages >= SCANNED_IMAGE_ONLY_SHARE

    def record(self, page_number: int, kind: str) -> None:
        """Record one page's kind (as returned by classify_page).

        Args:
            page_number: 1-based page number.
            kind: "text", "blank", "image_only" or "thin".
        """
        self.pages += 1
        if kind == "blank":
            self.blank.append(page_number)
        elif kind == "image_only":
            self.image_only.append(page_number)
        elif kind == "thin":
            self.thin.append(page_number)


def classify_page(page: Any, text: str | None = None) -> str:
    """Classify a pdfplumber page by its text layer.

    Args:
        page: A pdfplumber page.
        text: The page's extracted text, if already extracted (avoids a second pass).

    Returns:
        "text", "blank" (no text, no images), "image_only" (images, no text) or
        "thin" (images and fewer than TEXT_THIN_PAGE_CHARS characters).
    """
    content = (page.extract_text() or "" if text is None else text).strip()
    has_images = bool(getattr(page, "images", None))
    if not content:
        return "image_only" if has_images else "blank"
    if has_images and len(content) < TEXT_THIN_PAGE_CHARS:
        return "thin"
    return "text"


def text_coverage(path: Path) -> TextCoverage:
    """Check a PDF's text layer without chunking or indexing it.

    Args:
        path: PDF file.

    Returns:
        Its TextCoverage.

    Raises:
        RAGError: If the PDF can't be opened.
    """
    from quarq.exceptions import RAGError

    coverage = TextCoverage(source=path.name)
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                coverage.record(page.page_number, classify_page(page))
    except Exception as exc:
        raise RAGError(f"Failed to read PDF {path}: {exc}") from exc
    return coverage


def _infer_doc_type(filename: str) -> str:
    """Map a filename to a doc_type using substring rules.

    Args:
        filename: Basename of the PDF file (lowercase comparison).

    Returns:
        doc_type string matching rag-rules.md vocabulary.
    """
    lower = filename.lower()
    for substring, doc_type in _DOC_TYPE_RULES:
        if substring in lower:
            return doc_type
    return "macro"


def _chunk_text(
    text: str,
    page_num: int,
    chunk_size: int,
    chunk_overlap: int,
) -> list[tuple[str, int]]:
    """Split text into overlapping word-count chunks.

    Uses paragraph boundaries first, then falls back to word-level splitting.

    Args:
        text: Raw text to chunk.
        page_num: Source page number attached to all chunks.
        chunk_size: Maximum words per chunk.
        chunk_overlap: Word overlap between consecutive chunks.

    Returns:
        List of (chunk_text, page_num) tuples.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        paragraphs = [text.strip()] if text.strip() else []

    chunks: list[tuple[str, int]] = []
    current_words: list[str] = []

    for para in paragraphs:
        para_words = para.split()
        if not para_words:
            continue

        current_words.extend(para_words)

        while len(current_words) >= chunk_size:
            chunk_text = " ".join(current_words[:chunk_size])
            chunks.append((chunk_text, page_num))
            current_words = current_words[chunk_size - chunk_overlap:]

    # After a full chunk, the first chunk_overlap leftover words are already the tail of
    # that chunk. Emit the remainder only if it adds words of its own; otherwise it would
    # be a chunk entirely contained in the previous one.
    new_words = len(current_words) - (chunk_overlap if chunks else 0)
    if current_words and new_words > 0:
        chunk_text = " ".join(current_words)
        chunks.append((chunk_text, page_num))

    return chunks


def chunk_id(source: str, page: int, text: str) -> str:
    """Return the stable id of a chunk: sha256 of "source:page:text".

    Including the source and page means the same paragraph in two documents gets two
    ids. Re-indexing an unchanged file reproduces the same ids, so upserts don't duplicate.

    Args:
        source: Source filename.
        page: Page number.
        text: Chunk text.

    Returns:
        Hex sha256 digest.
    """
    return hashlib.sha256(f"{source}:{page}:{text}".encode()).hexdigest()


def _extract_pdf_date(metadata: dict, filename: str) -> str:
    """Extract a date string from PDF metadata or filename.

    Args:
        metadata: PDF metadata dict (may be empty).
        filename: Basename of the file as fallback.

    Returns:
        ISO-style date string or 'unknown'.
    """
    raw = metadata.get("CreationDate") or metadata.get("ModDate") or ""
    if raw and len(raw) >= 10:
        cleaned = raw.lstrip("D:").replace("'", "").replace(":", "")
        digits = "".join(c for c in cleaned if c.isdigit())
        if len(digits) >= 8:
            return f"{digits[:4]}-{digits[4:6]}-{digits[6:8]}"

    match = re.search(r"(20\d{2})", filename)
    if match:
        return match.group(1)

    return "unknown"


def load_pdf(
    path: Path,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    manifest: Manifest | None = None,
    coverage: TextCoverage | None = None,
) -> list[Document]:
    """Extract text from a PDF and return it as chunked Documents with metadata.

    Args:
        path: Absolute path to the PDF file.
        chunk_size: Maximum words per chunk (defaults to 512).
        chunk_overlap: Word overlap between consecutive chunks (defaults to 64).
        manifest: Optional corpus manifest. If it has an entry for this file, its
            period, publication date and doc_type override the inferred values.
        coverage: Optional TextCoverage to fill in the same pass: which pages are
            blank, image-only or thin. Pages without text are not chunked.

    Returns:
        List of Document instances, each with all five required metadata fields:
        source, doc_type, date, page, chunk_id (plus period_start and period_end
        when the manifest covers the file).

    Raises:
        RAGError: If the PDF cannot be opened or text cannot be extracted.
    """
    from quarq.exceptions import RAGError

    filename = path.name
    doc_type = _infer_doc_type(filename)
    entry = (manifest or {}).get(filename)
    overrides = entry.metadata() if entry is not None else {}

    try:
        with pdfplumber.open(path) as pdf:
            pdf_date = _extract_pdf_date(pdf.metadata or {}, filename)
            all_chunks: list[Document] = []

            if coverage is not None:
                coverage.source = filename
            for page in pdf.pages:
                page_num = page.page_number
                text = page.extract_text() or ""
                if coverage is not None:
                    coverage.record(page_num, classify_page(page, text))
                if not text.strip():
                    logger.debug("Page %d of %s yielded no text, skipping", page_num, filename)
                    continue

                for chunk_text, pnum in _chunk_text(text, page_num, chunk_size, chunk_overlap):
                    cid = chunk_id(filename, pnum, chunk_text)
                    doc = Document(
                        content=chunk_text,
                        metadata={
                            "source": filename,
                            "doc_type": doc_type,
                            "date": pdf_date,
                            "page": pnum,
                            "chunk_id": cid,
                            **overrides,
                        },
                    )
                    all_chunks.append(doc)

    except Exception as exc:
        raise RAGError(f"Failed to load PDF {path}: {exc}") from exc

    return all_chunks


def load_folder(
    folder: Path,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    manifest: Manifest | None = None,
    coverages: list[TextCoverage] | None = None,
) -> list[Document]:
    """Load and chunk all PDFs in a folder recursively.

    Args:
        folder: Path to search for PDF files.
        chunk_size: Maximum words per chunk.
        chunk_overlap: Word overlap between consecutive chunks.
        manifest: Optional corpus manifest, passed to load_pdf for each file.
        coverages: Optional list; one TextCoverage per loaded file is appended to it.

    Returns:
        Flat list of Document chunks from all found PDFs.
    """
    documents: list[Document] = []
    for pdf_path in sorted(folder.rglob("*")):
        if pdf_path.suffix.lower() != ".pdf":
            if pdf_path.is_file():
                logger.warning("Skipping non-PDF file: %s", pdf_path)
            continue
        try:
            coverage = TextCoverage(source=pdf_path.name) if coverages is not None else None
            docs = load_pdf(
                pdf_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap, manifest=manifest,
                coverage=coverage,
            )
            if coverages is not None and coverage is not None:
                coverages.append(coverage)
            documents.extend(docs)
            logger.info("Loaded %d chunks from %s", len(docs), pdf_path.name)
        except Exception as exc:
            logger.warning("Failed to load %s: %s", pdf_path, exc)

    return documents
