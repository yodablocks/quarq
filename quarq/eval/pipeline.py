"""Load a gold set, run it, and write both reports."""

from __future__ import annotations

from pathlib import Path

from quarq.config import QuarqConfig
from quarq.constants import DEFAULT_K_VALUES
from quarq.eval.dataset import load_gold
from quarq.eval.reporter import report_paths, to_json, to_markdown
from quarq.eval.runner import EvalResult, RetrieverLike, run_eval


def evaluate(
    retriever: RetrieverLike,
    cfg: QuarqConfig,
    *,
    dataset_path: Path,
    out_dir: Path,
    corpus_chunk_count: int,
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
    use_doc_type_filter: bool = False,
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

    Returns:
        (result, json_path, markdown_path).

    Raises:
        EvalError: On an invalid gold set or an existing report path.
        RAGError: Propagated from the retriever.
    """
    gold = load_gold(dataset_path)
    result = run_eval(
        retriever,
        gold,
        cfg,
        dataset_name=dataset_path.stem,
        corpus_chunk_count=corpus_chunk_count,
        k_values=k_values,
        use_doc_type_filter=use_doc_type_filter,
    )
    json_path, md_path = report_paths(result, out_dir)
    to_json(result, json_path)
    to_markdown(result, md_path)
    return result, json_path, md_path
