"""Gold-set models, loader, and writer for the retrieval eval."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

from quarq.exceptions import EvalError

Ref = tuple[str, int]

DRAFT_PROVENANCE = "synthetic-draft"
ACCEPTED_PROVENANCES: frozenset[str] = frozenset(
    {"synthetic-draft+human-accept", "hand-written"}
)


class GoldRef(BaseModel):
    """One correct (source, page) location for a question."""

    source: str
    page: int

    def key(self) -> Ref:
        """Return the (source, page) tuple used for matching.

        Returns:
            The (source, page) pair.
        """
        return (self.source, self.page)


class GoldItem(BaseModel):
    """A question with every page that correctly answers it."""

    id: str
    question: str = Field(min_length=1)
    gold: list[GoldRef] = Field(min_length=1)
    doc_type: str | None = None
    note: str = ""
    provenance: str = "synthetic-draft+human-accept"

    def gold_keys(self) -> set[Ref]:
        """Return the gold locations as a set of (source, page) tuples.

        Returns:
            Set of (source, page) pairs.
        """
        return {ref.key() for ref in self.gold}


def default_dataset_path() -> Path:
    """Return the path of the packaged v1 gold set.

    Returns:
        Path to quarq/eval/datasets/quarq_gold_v1.jsonl.
    """
    return Path(__file__).parent / "datasets" / "quarq_gold_v1.jsonl"


def load_gold(path: Path, require_accepted: bool = True) -> list[GoldItem]:
    """Load and validate a JSONL gold set.

    Args:
        path: JSONL file, one GoldItem per line. Blank lines are ignored.
        require_accepted: When True, reject rows whose provenance is not
            human-accepted, so unreviewed drafts can never be measured.

    Returns:
        Gold items in file order.

    Raises:
        EvalError: If the file is missing, a row is invalid, an id repeats,
            a draft is present while require_accepted is True, or the file
            holds no items.
    """
    if not path.exists():
        raise EvalError(f"Gold set not found: {path}")

    items: list[GoldItem] = []
    seen: set[str] = set()
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            item = GoldItem.model_validate_json(line)
        except ValidationError as exc:
            reason = exc.errors()[0]["msg"]
            raise EvalError(f"{path.name}:{lineno}: invalid gold row: {reason}") from exc
        if item.id in seen:
            raise EvalError(f"{path.name}:{lineno}: duplicate id {item.id!r}")
        if require_accepted and item.provenance not in ACCEPTED_PROVENANCES:
            raise EvalError(
                f"{path.name}:{lineno}: item {item.id!r} is not human-accepted "
                f"(provenance {item.provenance!r}). Review it and set provenance to "
                "'synthetic-draft+human-accept'."
            )
        seen.add(item.id)
        items.append(item)

    if not items:
        raise EvalError(f"{path.name}: no gold items found")
    return items


def write_gold(items: list[GoldItem], path: Path) -> None:
    """Write gold items as JSONL, refusing to overwrite an existing file.

    Args:
        items: Items to write, one per line.
        path: Destination file. Parent directories are created.

    Returns:
        None

    Raises:
        EvalError: If the destination already exists.
    """
    if path.exists():
        raise EvalError(f"Refusing to overwrite {path}: file already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [item.model_dump_json() for item in items]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
