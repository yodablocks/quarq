"""Load a gold set, run it, and write both reports."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from quarq.config import QuarqConfig
from quarq.constants import DEFAULT_K_VALUES, EVAL_UNKNOWN_REFS_SHOWN
from quarq.eval.dataset import GoldItem, Ref, load_gold, unknown_gold_refs
from quarq.eval.index_check import IndexCheck
from quarq.eval.reporter import report_paths, to_json, to_markdown
from quarq.eval.runner import EvalResult, RetrieverLike, run_eval
from quarq.exceptions import EvalError


def _check_known_refs(gold: list[GoldItem], known_refs: set[Ref]) -> None:
    """Raise EvalError if any gold ref is missing from known_refs.

    Args:
        gold: Loaded gold items to check.
        known_refs: (source, page) pairs present in the corpus.

    Raises:
        EvalError: If any gold ref is not in known_refs.
    """
    unknown = unknown_gold_refs(gold, known_refs)
    if not unknown:
        return
    shown = unknown[:EVAL_UNKNOWN_REFS_SHOWN]
    rest = len(unknown) - len(shown)
    lines = [f"{item_id}: {source} p{page}" for item_id, (source, page) in shown]
    message = "Gold refs missing from the corpus: " + "; ".join(lines)
    if rest:
        message += f"; and {rest} more"
    message += (
        ". Gold pages must use the PDF viewer's page index (1 = first physical page), "
        "not the printed page number."
    )
    raise EvalError(message)


def evaluate(
    retriever: RetrieverLike,
    cfg: QuarqConfig,
    *,
    dataset_path: Path,
    out_dir: Path,
    corpus_chunk_count: int,
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
    use_doc_type_filter: bool = False,
    known_refs: set[Ref] | None = None,
    index_checker: Callable[[list[GoldItem]], IndexCheck] | None = None,
) -> tuple[EvalResult, Path, Path]:
    """Run one full evaluation and write its JSON and Markdown reports.

    Args:
        retriever: Object with Retriever.retrieve's signature.
        cfg: Loaded quarq configuration.
        dataset_path: Accepted gold set (JSONL).
        out_dir: Directory for the timestamped reports.
        corpus_chunk_count: Chunks in the collection at run time.
        k_values: Cut-offs to report.
        use_doc_type_filter: Restrict each retrieval to the item's doc_type.
        known_refs: When given, every gold (source, page) ref must be in this
            set, or evaluation is refused before any retrieval or report write.
        index_checker: When given, called with the gold items; its IndexCheck
            (vector index vs exact search) is added to the result and reports.

    Returns:
        (result, json_path, markdown_path).

    Raises:
        EvalError: On an invalid gold set, a gold ref missing from known_refs,
            or an existing report path.
        RAGError: Propagated from the retriever.
    """
    gold = load_gold(dataset_path)
    if known_refs is not None:
        _check_known_refs(gold, known_refs)
    result = run_eval(
        retriever,
        gold,
        cfg,
        dataset_name=dataset_path.stem,
        corpus_chunk_count=corpus_chunk_count,
        k_values=k_values,
        use_doc_type_filter=use_doc_type_filter,
    )
    if index_checker is not None:
        result.index_check = index_checker(gold)
    json_path, md_path = report_paths(result, out_dir)
    to_json(result, json_path)
    to_markdown(result, md_path)
    return result, json_path, md_path
