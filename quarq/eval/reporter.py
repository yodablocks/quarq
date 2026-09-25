"""Render eval results as a rich table, a JSON file, and a Markdown report."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from rich.table import Table

from quarq.constants import EVAL_REPORT_REFS_SHOWN, EVAL_REPORT_WORST_N
from quarq.eval.index_check import IndexCheck
from quarq.eval.runner import EvalResult, PerQuestionResult
from quarq.exceptions import EvalError


def _stamp(generated_at: str) -> str:
    return datetime.fromisoformat(generated_at).strftime("%Y%m%dT%H%M%S")


def _refuse_overwrite(path: Path) -> None:
    if path.exists():
        raise EvalError(f"Refusing to overwrite {path}: file already exists")


def _md_cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def _refs(refs: list[tuple[str, int]]) -> str:
    return ", ".join(f"{source} p{page}" for source, page in refs) or "(none)"


def _provenance_summary(provenance_counts: dict[str, int]) -> str:
    return ", ".join(f"{prov} ({n})" for prov, n in provenance_counts.items())


def index_check_line(check: IndexCheck) -> str:
    """Summarise an IndexCheck in one line for the console.

    Args:
        check: The index check from an eval run.

    Returns:
        A one-line summary.
    """
    return (
        f"Index vs exact search: {check.questions_with_gap} / {check.n_questions} questions "
        f"miss a true neighbour in {check.candidates} candidates "
        f"({check.missed_chunks} chunks); final pages exact for "
        f"{check.exact_match_questions} / {check.n_questions}"
    )


def render_table(result: EvalResult) -> Table:
    """Build a rich table of aggregate metrics per k.

    Args:
        result: A completed eval run.

    Returns:
        A rich Table ready for console.print.
    """
    filt = "on" if result.use_doc_type_filter else "off"
    table = Table(
        title=f"Retrieval eval: {result.dataset_name} "
              f"({result.n_questions} questions, doc_type filter {filt})",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("k", justify="right")
    table.add_column("Precision@k", justify="right")
    table.add_column("Recall@k", justify="right")
    table.add_column("Hit@k", justify="right")
    for k in result.k_values:
        agg = result.aggregate
        table.add_row(
            str(k),
            f"{agg[f'precision@{k}']:.3f}",
            f"{agg[f'recall@{k}']:.3f}",
            f"{agg[f'hit@{k}']:.3f}",
        )
    table.caption = f"MRR {result.aggregate['mrr']:.3f}"
    return table


def report_paths(result: EvalResult, out_dir: Path) -> tuple[Path, Path]:
    """Return timestamped JSON and Markdown paths for a run.

    Args:
        result: A completed eval run.
        out_dir: Directory the reports go in.

    Returns:
        (json_path, markdown_path).
    """
    stem = f"eval_report_{_stamp(result.generated_at)}"
    return out_dir / f"{stem}.json", out_dir / f"{stem}.md"


def to_json(result: EvalResult, path: Path) -> Path:
    """Write the full result, including per-question detail, as JSON.

    Args:
        result: A completed eval run.
        path: Destination file; parents are created.

    Returns:
        The path written.

    Raises:
        EvalError: If the path already exists or cannot be written.
    """
    _refuse_overwrite(path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(asdict(result), indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except OSError as exc:
        raise EvalError(f"Cannot write report {path}: {exc}") from exc
    return path


def _index_check_markdown(result: EvalResult) -> list[str]:
    check = result.index_check
    if check is None:
        return []
    return [
        "## Index check",
        "",
        "The vector index is approximate. Compared with exact search over the same chunks:",
        "",
        f"- Raw index: {check.questions_with_gap} / {check.n_questions} questions miss a true "
        f"nearest neighbour among {check.candidates} candidates ({check.missed_chunks} "
        "chunks in total; ties between near-duplicates not counted).",
        f"- End to end: the retrieved pages match exact search for "
        f"{check.exact_match_questions} / {check.n_questions} questions.",
        "",
    ]


def _worst(result: EvalResult, worst_n: int) -> list[PerQuestionResult]:
    max_k = max(result.k_values)
    ranked = sorted(
        result.per_question,
        key=lambda q: (q.metrics[f"recall@{max_k}"], q.metrics["mrr"]),
    )
    return ranked[:worst_n]


def to_markdown(result: EvalResult, path: Path, worst_n: int = EVAL_REPORT_WORST_N) -> Path:
    """Write a shareable Markdown report.

    Args:
        result: A completed eval run.
        path: Destination file; parents are created.
        worst_n: How many lowest-recall questions to list for error analysis.

    Returns:
        The path written.

    Raises:
        EvalError: If the path already exists or cannot be written.
    """
    _refuse_overwrite(path)
    max_k = max(result.k_values)
    filt = "on" if result.use_doc_type_filter else "off"
    lines = [
        f"# quarq retrieval eval: {result.dataset_name}",
        "",
        f"Generated: {result.generated_at}  ",
        f"Questions: {result.n_questions}  ",
        f"Corpus chunks: {result.corpus_chunk_count}  ",
        f"doc_type filter: {filt}  ",
        f"Provenance: {_provenance_summary(result.provenance_counts)}",
        "",
        "## Config",
        "",
        "| setting | value |",
        "|---|---|",
        *[f"| {key} | {value} |" for key, value in result.config_snapshot.items()],
        "",
        "## Results",
        "",
        "| k | Precision@k | Recall@k | Hit@k |",
        "|---|---|---|---|",
    ]
    for k in result.k_values:
        agg = result.aggregate
        lines.append(
            f"| {k} | {agg[f'precision@{k}']:.3f} | {agg[f'recall@{k}']:.3f} "
            f"| {agg[f'hit@{k}']:.3f} |"
        )
    lines += [
        "",
        f"MRR: {result.aggregate['mrr']:.3f}",
        "",
        *_index_check_markdown(result),
        f"## Worst questions (lowest recall@{max_k}, then MRR)",
        "",
        f"| id | question | recall@{max_k} | gold | top retrieved |",
        "|---|---|---|---|---|",
    ]
    for q in _worst(result, worst_n):
        top_retrieved = _md_cell(_refs(q.retrieved[:EVAL_REPORT_REFS_SHOWN]))
        lines.append(
            f"| {q.id} | {_md_cell(q.question)} | {q.metrics[f'recall@{max_k}']:.3f} "
            f"| {_md_cell(_refs(q.gold))} | {top_retrieved} |"
        )
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except OSError as exc:
        raise EvalError(f"Cannot write report {path}: {exc}") from exc
    return path
