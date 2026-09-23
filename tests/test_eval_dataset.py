"""Tests for quarq.eval.dataset."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from quarq.eval.dataset import (
    DRAFT_PROVENANCE,
    GoldItem,
    GoldRef,
    default_dataset_path,
    load_gold,
    write_gold,
)
from quarq.exceptions import EvalError


def _row(**overrides: object) -> str:
    row: dict[str, object] = {
        "id": "q001",
        "question": "What risk did the ECB flag for equity valuations?",
        "gold": [{"source": "ecb.pdf", "page": 12}],
        "doc_type": "ecb_fsr",
        "note": "fig 1.4",
        "provenance": "synthetic-draft+human-accept",
    }
    row.update(overrides)
    return json.dumps(row)


def test_load_gold_parses_valid_rows(tmp_path: Path) -> None:
    path = tmp_path / "gold.jsonl"
    path.write_text(_row() + "\n\n" + _row(id="q002") + "\n", encoding="utf-8")

    items = load_gold(path)

    assert [i.id for i in items] == ["q001", "q002"]
    assert items[0].gold_keys() == {("ecb.pdf", 12)}
    assert items[0].doc_type == "ecb_fsr"


def test_load_gold_rejects_malformed_json_with_line_number(tmp_path: Path) -> None:
    path = tmp_path / "gold.jsonl"
    path.write_text(_row() + "\n{not json\n", encoding="utf-8")

    with pytest.raises(EvalError, match=r"gold\.jsonl:2"):
        load_gold(path)


def test_load_gold_rejects_empty_gold_list(tmp_path: Path) -> None:
    path = tmp_path / "gold.jsonl"
    path.write_text(_row(gold=[]) + "\n", encoding="utf-8")

    with pytest.raises(EvalError, match=r"gold\.jsonl:1"):
        load_gold(path)


def test_load_gold_rejects_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "gold.jsonl"
    path.write_text("\n", encoding="utf-8")

    with pytest.raises(EvalError, match="no gold items"):
        load_gold(path)


def test_load_gold_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(EvalError, match="not found"):
        load_gold(tmp_path / "missing.jsonl")


def test_load_gold_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = tmp_path / "gold.jsonl"
    path.write_text(_row() + "\n" + _row() + "\n", encoding="utf-8")

    with pytest.raises(EvalError, match="duplicate id 'q001'"):
        load_gold(path)


def test_load_gold_refuses_unaccepted_drafts_by_default(tmp_path: Path) -> None:
    path = tmp_path / "gold.jsonl"
    path.write_text(_row(provenance=DRAFT_PROVENANCE) + "\n", encoding="utf-8")

    with pytest.raises(EvalError, match="not human-accepted"):
        load_gold(path)
    assert load_gold(path, require_accepted=False)[0].provenance == DRAFT_PROVENANCE


def test_write_gold_round_trips_and_refuses_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "out" / "drafts.jsonl"
    item = GoldItem(
        id="draft001",
        question="Q?",
        gold=[GoldRef(source="a.pdf", page=3)],
        provenance=DRAFT_PROVENANCE,
    )

    write_gold([item], path)

    assert load_gold(path, require_accepted=False) == [item]
    with pytest.raises(EvalError, match="already exists"):
        write_gold([item], path)


def test_default_dataset_path_points_at_packaged_v1() -> None:
    path = default_dataset_path()
    assert path.name == "quarq_gold_v1.jsonl"
    assert path.parent.name == "datasets"
