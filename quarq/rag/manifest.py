"""Corpus manifest: per-document metadata kept next to the PDFs.

A PDF's own metadata says when the file was created, not what period it covers:
an annual report on 2024 is created in 2025, and a regenerated stability review
can carry a date months after its edition. The manifest records the facts a
person can check (period covered, publication date, and optionally doc_type),
so retrieval can use them instead of guesses.

Format (TOML, one table per document, keyed by filename)::

    [[document]]
    source = "bdf_rapport_annuel_2024.pdf"
    doc_type = "bdf_fsr"          # optional, overrides the filename rule
    published = 2025-03-17        # optional, becomes the chunk `date`
    period_start = 2024-01-01     # required
    period_end = 2024-12-31       # required
    note = "free text for reviewers, not indexed"
"""

from __future__ import annotations

import logging
import tomllib
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from quarq.constants import CORPUS_MANIFEST_FILENAME, DOC_TYPES
from quarq.exceptions import RAGError

if TYPE_CHECKING:
    from quarq.rag.store import VectorStore

logger = logging.getLogger(__name__)


class ManifestEntry(BaseModel):
    """Checked metadata for one source document."""

    model_config = ConfigDict(extra="forbid")

    source: str = Field(min_length=1)
    period_start: date
    period_end: date
    published: date | None = None
    doc_type: str | None = None
    note: str = ""

    @field_validator("doc_type")
    @classmethod
    def _known_doc_type(cls, value: str | None) -> str | None:
        if value is not None and value not in DOC_TYPES:
            raise ValueError(f"unknown doc_type {value!r}; allowed: {sorted(DOC_TYPES)}")
        return value

    @model_validator(mode="after")
    def _period_in_order(self) -> ManifestEntry:
        if self.period_start > self.period_end:
            raise ValueError(
                f"period_start {self.period_start} is after period_end {self.period_end}"
            )
        return self

    def metadata(self) -> dict[str, str]:
        """Return the chunk metadata this entry sets, as ChromaDB-safe strings.

        Only fields that are set are returned, so an entry without `published`
        or `doc_type` leaves the loader's values for those keys in place.

        Returns:
            Metadata keys to merge into every chunk of the source.
        """
        meta: dict[str, str] = {}
        if self.doc_type is not None:
            meta["doc_type"] = self.doc_type
        if self.published is not None:
            meta["date"] = self.published.isoformat()
        meta["period_start"] = self.period_start.isoformat()
        meta["period_end"] = self.period_end.isoformat()
        return meta


Manifest = dict[str, ManifestEntry]


def load_manifest(path: Path) -> Manifest:
    """Parse and validate a manifest file.

    Args:
        path: Path to a manifest TOML file.

    Returns:
        Entries keyed by source filename.

    Raises:
        RAGError: If the file can't be read or parsed, or an entry is invalid. The
            message names the source (or entry number) and the field.
    """
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise RAGError(f"Cannot read manifest {path}: {exc}") from exc

    unknown = set(data) - {"document"}
    if unknown:
        raise RAGError(
            f"{path.name}: unknown top-level key(s) {sorted(unknown)}; "
            "expected only [[document]] tables"
        )

    manifest: Manifest = {}
    for index, raw in enumerate(data.get("document", []), start=1):
        label = raw.get("source") if isinstance(raw, dict) else None
        where = f"source {label!r}" if label else f"entry {index}"
        try:
            entry = ManifestEntry.model_validate(raw)
        except ValidationError as exc:
            err = exc.errors()[0]
            loc = ".".join(str(p) for p in err["loc"])
            reason = f"{loc}: {err['msg']}" if loc else err["msg"]
            raise RAGError(f"{path.name}: {where}: {reason}") from exc
        if entry.source in manifest:
            raise RAGError(f"{path.name}: duplicate source {entry.source!r}")
        manifest[entry.source] = entry
    return manifest


def find_manifest(target: Path) -> Path | None:
    """Locate the manifest for a folder, or for a single PDF (its parent folder).

    Args:
        target: A folder being indexed, or a PDF file.

    Returns:
        The manifest path if one exists, else None.
    """
    folder = target.parent if target.is_file() else target
    candidate = folder / CORPUS_MANIFEST_FILENAME
    return candidate if candidate.is_file() else None


@dataclass
class ApplyReport:
    """What apply_manifest changed and what it couldn't match."""

    updated: dict[str, int] = field(default_factory=dict)
    not_indexed: list[str] = field(default_factory=list)
    not_in_manifest: list[str] = field(default_factory=list)


def apply_manifest(store: VectorStore, manifest: Manifest) -> ApplyReport:
    """Write manifest metadata onto chunks already in the index, without re-embedding.

    Running it twice gives the same result.

    Args:
        store: The vector store to update.
        manifest: Entries from load_manifest.

    Returns:
        Chunks updated per source, manifest sources missing from the index, and
        indexed sources missing from the manifest.

    Raises:
        RAGError: If the store can't be read or updated.
    """
    indexed = store.list_sources()
    report = ApplyReport(
        not_indexed=sorted(set(manifest) - indexed, key=str.lower),
        not_in_manifest=sorted(indexed - set(manifest), key=str.lower),
    )
    for source in sorted(set(manifest) & indexed, key=str.lower):
        fields: dict[str, Any] = dict(manifest[source].metadata())
        report.updated[source] = store.update_source_metadata(source, fields)
    return report
