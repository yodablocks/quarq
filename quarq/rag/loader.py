"""PDF text extraction and chunking with required metadata attachment."""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import pdfplumber

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

    if current_words:
        chunk_text = " ".join(current_words)
        chunks.append((chunk_text, page_num))

    return chunks


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


def load_pdf(path: Path, chunk_size: int = 512, chunk_overlap: int = 64) -> list[Document]:
    """Extract text from a PDF and return it as chunked Documents with metadata.

    Args:
        path: Absolute path to the PDF file.
        chunk_size: Maximum words per chunk (defaults to 512).
        chunk_overlap: Word overlap between consecutive chunks (defaults to 64).

    Returns:
        List of Document instances, each with all five required metadata fields:
        source, doc_type, date, page, chunk_id.

    Raises:
        RAGError: If the PDF cannot be opened or text cannot be extracted.
    """
    from quarq.exceptions import RAGError

    filename = path.name
    doc_type = _infer_doc_type(filename)

    try:
        with pdfplumber.open(path) as pdf:
            pdf_date = _extract_pdf_date(pdf.metadata or {}, filename)
            all_chunks: list[Document] = []

            for page in pdf.pages:
                page_num = page.page_number
                text = page.extract_text() or ""
                if not text.strip():
                    logger.debug("Page %d of %s yielded no text, skipping", page_num, filename)
                    continue

                for chunk_text, pnum in _chunk_text(text, page_num, chunk_size, chunk_overlap):
                    chunk_id = hashlib.sha256(
                        f"{filename}:{pnum}:{chunk_text}".encode("utf-8")
                    ).hexdigest()
                    doc = Document(
                        content=chunk_text,
                        metadata={
                            "source": filename,
                            "doc_type": doc_type,
                            "date": pdf_date,
                            "page": pnum,
                            "chunk_id": chunk_id,
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
) -> list[Document]:
    """Load and chunk all PDFs in a folder recursively.

    Args:
        folder: Path to search for PDF files.
        chunk_size: Maximum words per chunk.
        chunk_overlap: Word overlap between consecutive chunks.

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
            docs = load_pdf(pdf_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            documents.extend(docs)
            logger.info("Loaded %d chunks from %s", len(docs), pdf_path.name)
        except Exception as exc:
            logger.warning("Failed to load %s: %s", pdf_path, exc)

    return documents
