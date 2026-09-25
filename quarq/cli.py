"""quarq CLI entry point.

Renders the launch screen with live connection status using rich.
Handles argparse subcommands: (default), status, version.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from quarq import __version__
from quarq.config import load_config
from quarq.constants import RAG_COLLECTION_NAME
from quarq.status import StatusResult, run_status_checks

console = Console()


def _status_table(result: StatusResult, lmstudio_url: str) -> Table:
    """Build a rich Table from a StatusResult.

    Args:
        result: Populated StatusResult from run_status_checks().
        lmstudio_url: LM Studio base URL from config, shown in the table.

    Returns:
        Formatted rich Table ready to render.
    """
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("indicator", style="bold", width=6)
    table.add_column("service", width=14)
    table.add_column("detail", width=30)
    table.add_column("url", style="dim")

    # LM Studio row
    if result.lmstudio_online:
        table.add_row(
            Text("[ok]", style="bold green"),
            "LM Studio",
            result.lmstudio_active_model or "connected",
            lmstudio_url,
        )
    else:
        table.add_row(
            Text("[--]", style="bold red"),
            "LM Studio",
            "offline",
            lmstudio_url,
        )

    # FRED row
    fred_colors = {"connected": "green", "no key": "yellow", "offline": "red"}
    fred_indicators = {"connected": "[ok]", "no key": "[~~]", "offline": "[--]"}
    fred_labels = {"connected": "connected", "no key": "no key set", "offline": "offline"}
    table.add_row(
        Text(fred_indicators[result.fred_status], style=f"bold {fred_colors[result.fred_status]}"),
        "FRED API",
        fred_labels[result.fred_status],
        "",
    )

    # ECB row
    if result.ecb_online:
        table.add_row(Text("[ok]", style="bold green"), "ECB", "connected", "")
    else:
        table.add_row(Text("[--]", style="bold red"), "ECB", "offline", "")

    # RAG corpus row
    if result.corpus_docs > 0:
        table.add_row(
            Text("[ok]", style="bold green"),
            "RAG corpus",
            f"{result.corpus_docs} docs  {result.corpus_chunks} chunks",
            "",
        )
    else:
        table.add_row(Text("[--]", style="bold red"), "RAG corpus", "empty", "")

    return table


def _menu_table() -> Table:
    """Build the menu options table.

    Returns:
        Formatted rich Table with menu entries.
    """
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("key", style="bold cyan", width=6)
    table.add_column("action")

    table.add_row("[1]", Text("Generate report  (Phase 3)", style="dim"))
    table.add_row("[2]", Text("Query RAG  (Phase 4)", style="dim"))
    table.add_row("[3]", Text("Configure  (Phase 3)", style="dim"))
    table.add_row("[4]", "Status  refresh")
    table.add_row("[q]", "Quit")

    return table


def _render_launch_screen(result: StatusResult, lmstudio_url: str) -> None:
    """Render the full launch screen panel.

    Args:
        result: Populated StatusResult.
        lmstudio_url: LM Studio base URL from config.
    """
    title = Text()
    title.append("QUARQ", style="bold white")
    title.append(f"  v{result.quarq_version}", style="dim")
    title.append("  --  French institutional portfolio analytics", style="italic dim")

    console.print(Panel(title, expand=False))
    console.print(_status_table(result, lmstudio_url))
    console.print()
    console.print("  Select:")
    console.print(_menu_table())


def _cmd_status() -> None:
    """Run status checks and print result as a rich table, then exit."""
    cfg = load_config()
    result = asyncio.run(run_status_checks(cfg))

    table = Table(title="quarq status", show_header=True, header_style="bold cyan")
    table.add_column("Check")
    table.add_column("Status")

    lm_status = "online" if result.lmstudio_online else "offline"
    table.add_row("LM Studio", lm_status)
    if result.lmstudio_active_model:
        table.add_row("  active model", result.lmstudio_active_model)
    table.add_row("FRED", result.fred_status)
    table.add_row("ECB", "online" if result.ecb_online else "offline")
    table.add_row(
        "RAG corpus",
        f"{result.corpus_docs} docs / {result.corpus_chunks} chunks",
    )
    table.add_row("Config", str(result.config_path))
    table.add_row("Version", result.quarq_version)

    console.print(table)


def _launch_loop() -> None:
    """Run the interactive launch screen loop."""
    cfg = load_config()

    while True:
        console.clear()
        with console.status("[bold cyan]Checking connections...", spinner="dots"):
            result = asyncio.run(run_status_checks(cfg))

        _render_launch_screen(result, cfg.lmstudio.url)
        console.print()

        try:
            choice = input("  > ").strip().lower()
        except EOFError:
            break

        if choice in ("q", "quit", "exit"):
            break
        elif choice == "4":
            continue
        elif choice in ("1", "2", "3"):
            console.print(
                "\n  [yellow]Not yet implemented — coming in Phase 3[/yellow]\n"
            )
            input("  Press Enter to return...")
        else:
            console.print("\n  [dim]Unknown option. Try 1-4 or q.[/dim]\n")
            input("  Press Enter to continue...")


def _cmd_query(question: str, doc_type: str | None, k: int) -> None:
    """Run a RAG query and print the answer with citations.

    Args:
        question: Natural language question to answer.
        doc_type: Optional doc_type filter (e.g. 'ecb_fsr').
        k: Number of chunks to retrieve.
    """
    from quarq.rag.embedder import Embedder
    from quarq.rag.generator import answer, format_citations
    from quarq.rag.retriever import Retriever
    from quarq.rag.store import VectorStore

    cfg = load_config()
    store = VectorStore(cfg)

    if store.count() == 0:
        console.print(
            Panel(
                "[yellow]RAG corpus is empty. Run: [bold]quarq rag add ./docs/[/bold] "
                "to index documents.[/yellow]",
                title="No documents indexed",
                border_style="yellow",
            )
        )
        return

    embedder = Embedder(model_name=cfg.embedder.model)
    retriever = Retriever(store=store, embedder=embedder, cfg=cfg)

    with console.status("[bold cyan]Retrieving relevant documents...", spinner="dots"):
        chunks = retriever.retrieve(question, k=k, doc_type=doc_type)

    if not chunks:
        console.print(
            Panel(
                "[yellow]No relevant documents found above similarity threshold.[/yellow]\n"
                "Try a different question or lower --k.",
                border_style="yellow",
            )
        )
        return

    with console.status("[bold cyan]Generating answer...", spinner="dots"):
        result = answer(question, chunks, cfg)

    console.print(Panel(result.answer, title="[bold]Answer[/bold]", border_style="green"))

    citation_table = Table(title="Sources", show_header=True, header_style="bold cyan")
    citation_table.add_column("Source")
    citation_table.add_column("Page", justify="right")
    citation_table.add_column("Similarity", justify="right")
    for chunk in chunks:
        citation_table.add_row(chunk.source, str(chunk.page), f"{chunk.similarity:.2f}")
    console.print(citation_table)

    console.print(f"[dim]Model: {result.model}  Backend: {result.backend}  "
                  f"Latency: {result.latency_ms}ms[/dim]")


def _cmd_rag_status() -> None:
    """Print RAG corpus statistics."""
    from quarq.rag.store import VectorStore

    cfg = load_config()
    store = VectorStore(cfg)
    chunk_count = store.count()
    source_count = store.count_sources()

    table = Table(title="RAG Corpus Status", show_header=True, header_style="bold cyan")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Chunks indexed", str(chunk_count))
    table.add_row("Source documents", str(source_count))
    table.add_row("Collection", RAG_COLLECTION_NAME)
    table.add_row("Chroma path", cfg.rag.chroma_path)
    console.print(table)


def _cmd_rag_add(path_str: str) -> None:
    """Index a file or folder into the RAG corpus.

    Args:
        path_str: Path to a PDF file or folder of PDFs.
    """
    from rich.progress import Progress, SpinnerColumn, TextColumn

    from quarq.exceptions import RAGError
    from quarq.rag.embedder import Embedder
    from quarq.rag.loader import load_folder, load_pdf
    from quarq.rag.manifest import find_manifest, load_manifest
    from quarq.rag.store import VectorStore

    cfg = load_config()
    target = Path(path_str).expanduser().resolve()

    if not target.exists():
        console.print(f"[red]Path does not exist: {target}[/red]")
        sys.exit(1)

    manifest_path = find_manifest(target)
    try:
        manifest = load_manifest(manifest_path) if manifest_path else None
    except RAGError as exc:
        console.print(Panel(f"[red]{exc}[/red]", title="Invalid manifest", border_style="red"))
        sys.exit(1)
    if manifest_path and manifest is not None:
        console.print(f"[dim]Using manifest {manifest_path} ({len(manifest)} documents)[/dim]")

    with console.status("[bold cyan]Loading documents...", spinner="dots"):
        if target.is_file():
            documents = load_pdf(target, chunk_size=cfg.rag.chunk_size,
                                 chunk_overlap=cfg.rag.chunk_overlap, manifest=manifest)
        else:
            documents = load_folder(target, chunk_size=cfg.rag.chunk_size,
                                    chunk_overlap=cfg.rag.chunk_overlap, manifest=manifest)

    if not documents:
        console.print("[yellow]No documents found to index.[/yellow]")
        return

    console.print(f"Loaded [bold]{len(documents)}[/bold] chunks from {target}")

    embedder = Embedder(model_name=cfg.embedder.model)
    store = VectorStore(cfg)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Embedding and indexing...", total=None)
        embeddings = embedder.embed([doc.content for doc in documents])
        added = store.upsert(documents, embeddings)
        progress.update(task, completed=True)

    console.print(
        f"[green]Done.[/green] Added [bold]{added}[/bold] new chunks "
        f"({len(documents) - added} duplicates skipped). "
        f"Corpus now has [bold]{store.count()}[/bold] chunks."
    )


def _cmd_rag_manifest(path_str: str) -> int:
    """Apply a corpus manifest to chunks already in the index, without re-embedding.

    Args:
        path_str: The manifest file, or the folder that contains it.

    Returns:
        Process exit code (0 on success).
    """
    from quarq.constants import CORPUS_MANIFEST_FILENAME
    from quarq.exceptions import RAGError
    from quarq.rag.manifest import apply_manifest, find_manifest, load_manifest
    from quarq.rag.store import VectorStore

    target = Path(path_str).expanduser().resolve()
    manifest_path = target if target.is_file() else find_manifest(target)
    if manifest_path is None:
        console.print(f"[red]No {CORPUS_MANIFEST_FILENAME} found at {target}[/red]")
        return 1

    try:
        manifest = load_manifest(manifest_path)
        report = apply_manifest(VectorStore(load_config()), manifest)
    except RAGError as exc:
        console.print(Panel(f"[red]{exc}[/red]", title="Manifest failed", border_style="red"))
        return 1

    table = Table(title=f"Manifest applied: {manifest_path.name}")
    table.add_column("Source")
    table.add_column("Chunks updated", justify="right")
    for source, n in report.updated.items():
        table.add_row(source, str(n))
    console.print(table)
    if report.not_indexed:
        missing = ", ".join(report.not_indexed)
        console.print(f"[yellow]In the manifest but not indexed: {missing}[/yellow]")
    if report.not_in_manifest:
        unlisted = ", ".join(report.not_in_manifest)
        console.print(f"[yellow]Indexed but missing from the manifest: {unlisted}[/yellow]")
    return 0


def _cmd_report(
    portfolio_path: str,
    fmt: str,
    output: str | None,
    include_narrative: bool,
    open_browser: bool,
) -> None:
    """Generate a portfolio report from a TOML file.

    Args:
        portfolio_path: Path to the portfolio TOML file.
        fmt: Output format: 'html', 'json', or 'pdf'.
        output: Output file path (default: quarq_report_YYYY-MM-DD.html).
        include_narrative: Whether to generate the LLM narrative paragraph.
        open_browser: Whether to open the output file in the browser after generation.
    """
    import webbrowser
    from datetime import date

    from quarq.api import metrics as m
    from quarq.ingest.equity import fetch_portfolio
    from quarq.ingest.fred import get_risk_free_rate
    from quarq.portfolio import load_portfolio
    from quarq.report.charts import (
        correlation_heatmap,
        cumulative_returns_line,
        drawdown_area,
        rolling_sharpe,
        weight_treemap,
    )
    from quarq.report.renderer import render_html, render_pdf

    import pandas as pd
    from quarq.api.models import MetricsResponse
    from quarq.exceptions import ConfigError, ProviderError, QuarqError

    cfg = load_config()

    try:
        spec = load_portfolio(Path(portfolio_path).expanduser().resolve())
    except ConfigError as exc:
        console.print(f"[red]Portfolio TOML error: {exc}[/red]")
        sys.exit(1)

    console.print(f"Loaded portfolio: [bold]{spec.name}[/bold] ({len(spec.tickers)} tickers)")

    with console.status("[cyan]Fetching price data...", spinner="dots"):
        try:
            prices = fetch_portfolio(
                tickers=spec.tickers,
                start=spec.start,
                end=spec.end,
                benchmark=spec.benchmark,
            )
        except ProviderError as exc:
            console.print(f"[red]Data fetch failed: {exc}[/red]")
            sys.exit(1)

    rfr = get_risk_free_rate(cfg)
    portfolio_returns = m.compute_portfolio_returns(prices, spec.tickers, spec.weights)
    bm_df = prices.get(spec.benchmark)
    benchmark_returns = bm_df["value"].pct_change().dropna() if bm_df is not None else None
    beta_val = m.beta(portfolio_returns, benchmark_returns) if benchmark_returns is not None else None
    alpha_val = (
        m.alpha(portfolio_returns, benchmark_returns, beta_val, rfr)
        if benchmark_returns is not None else None
    )

    metrics = MetricsResponse(
        tickers=spec.tickers,
        weights=spec.weights,
        start=spec.start,
        end=spec.end,
        benchmark=spec.benchmark,
        sharpe_ratio=m.sharpe_ratio(portfolio_returns, rfr),
        max_drawdown=m.max_drawdown(portfolio_returns),
        cagr=m.cagr(portfolio_returns),
        volatility=m.volatility(portfolio_returns),
        var_95=m.var_95(portfolio_returns),
        beta=beta_val,
        alpha=alpha_val,
        risk_free_rate=rfr,
    )

    returns_df_parts = []
    for ticker in spec.tickers:
        df = prices.get(ticker)
        if df is not None:
            s = df["value"].pct_change().dropna().rename(ticker)
            returns_df_parts.append(s)
    if len(returns_df_parts) > 1:
        returns_df = pd.concat(returns_df_parts, axis=1).dropna()
    elif returns_df_parts:
        returns_df = returns_df_parts[0].to_frame()
    else:
        returns_df = pd.DataFrame()

    bench_for_chart = benchmark_returns if benchmark_returns is not None else portfolio_returns

    figures = {
        "cumulative_returns": cumulative_returns_line(portfolio_returns, bench_for_chart, label=spec.name),
        "drawdown": drawdown_area(portfolio_returns),
        "rolling_sharpe": rolling_sharpe(portfolio_returns, rfr=rfr),
        "correlation_heatmap": correlation_heatmap(returns_df) if not returns_df.empty else None,
        "weight_treemap": weight_treemap(dict(zip(spec.tickers, spec.weights)), spec.sleeve_map),
    }

    narrative: str | None = None
    if include_narrative:
        with console.status("[cyan]Generating narrative...", spinner="dots"):
            narrative = m.generate_narrative(metrics, cfg) or None
        if not narrative:
            console.print(
                "[yellow]Warning: LM Studio unavailable — generating report without narrative.[/yellow]"
            )

    report_date = date.today().isoformat()
    html = render_html(
        portfolio_name=spec.name,
        metrics=metrics,
        figures={k: v for k, v in figures.items() if v is not None},
        narrative=narrative,
        benchmark=spec.benchmark,
        period_start=spec.start.isoformat(),
        period_end=spec.end.isoformat(),
        currency=spec.currency,
        report_date=report_date,
    )

    default_ext = "pdf" if fmt == "pdf" else "html"
    out_path = Path(output) if output else Path(f"quarq_report_{report_date}.{default_ext}")

    if fmt == "pdf":
        try:
            render_pdf(html, out_path)
            console.print(f"[green]PDF saved to:[/green] {out_path}")
        except QuarqError as exc:
            console.print(f"[yellow]Warning: {exc}[/yellow]")
            html_path = out_path.with_suffix(".html")
            html_path.write_text(html, encoding="utf-8")
            console.print(f"[green]HTML fallback saved to:[/green] {html_path}")
            out_path = html_path
    elif fmt == "json":
        import plotly.io as pio
        import json
        payload = {
            "metrics": metrics.model_dump(mode="json"),
            "charts": {k: pio.to_json(v) if v is not None else None for k, v in figures.items()},
        }
        out_path = out_path.with_suffix(".json") if output is None else out_path
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        console.print(f"[green]JSON saved to:[/green] {out_path}")
    else:
        out_path.write_text(html, encoding="utf-8")
        console.print(f"[green]Report saved to:[/green] {out_path}")

    if open_browser:
        webbrowser.open(out_path.resolve().as_uri())


def _cmd_config_set_lmstudio_url(url: str) -> None:
    """Update lmstudio.url in ~/.quarq/config.toml and print confirmation.

    Args:
        url: New LM Studio base URL, e.g. http://192.168.1.101:1234/v1.
    """
    from quarq.config import save_config

    cfg = load_config()
    old_url = cfg.lmstudio.url
    cfg.lmstudio.url = url
    save_config(cfg)
    console.print(f"[green]LM Studio URL updated.[/green]")
    console.print(f"  [dim]was:[/dim] {old_url}")
    console.print(f"  [dim]now:[/dim] {url}")


def _cmd_serve(host: str, port: int, reload: bool) -> None:
    """Start the quarq FastAPI server with uvicorn.

    Args:
        host: Bind address (default 127.0.0.1).
        port: TCP port (default 8000).
        reload: Enable auto-reload for development.
    """
    import uvicorn

    panel_content = (
        f"[bold white]quarq API server starting[/bold white]\n"
        f"http://{host}:{port}\n"
        f"Docs: http://{host}:{port}/docs\n"
        f"[dim]Press Ctrl+C to stop[/dim]"
    )
    console.print(Panel(panel_content, title="quarq serve", border_style="cyan"))
    uvicorn.run("quarq.api.app:app", host=host, port=port, reload=reload)


def _cmd_eval(dataset: str | None, k: str, out: str, doc_type_filter: bool) -> int:
    """Run the retrieval eval against the indexed corpus and write reports.

    Args:
        dataset: Gold-set JSONL path, or None for the packaged v1 set.
        k: Comma-separated k values, e.g. "1,3,5".
        out: Output directory for the timestamped reports.
        doc_type_filter: Restrict each retrieval to the item's doc_type.

    Returns:
        0 on success, 1 on a handled failure.
    """
    from pathlib import Path

    from quarq.eval.dataset import default_dataset_path
    from quarq.eval.index_check import check_index
    from quarq.eval.pipeline import evaluate
    from quarq.eval.reporter import index_check_line, render_table
    from quarq.eval.runner import parse_k_values
    from quarq.exceptions import EvalError, RAGError
    from quarq.rag.embedder import Embedder
    from quarq.rag.exact import ExactStore
    from quarq.rag.retriever import Retriever
    from quarq.rag.store import VectorStore

    cfg = load_config()
    try:
        store = VectorStore(cfg)
        chunk_count = store.count()
        if chunk_count == 0:
            console.print(
                Panel(
                    "[yellow]RAG corpus is empty. Run: [bold]quarq rag add ./docs/[/bold] "
                    "before evaluating.[/yellow]",
                    title="No documents indexed",
                    border_style="yellow",
                )
            )
            return 1
        embedder = Embedder(model_name=cfg.embedder.model)
        retriever = Retriever(store=store, embedder=embedder, cfg=cfg)
        k_values = parse_k_values(k)
        known_refs = {(c.source, int(c.page)) for c in store.list_chunks()}
        dataset_path = Path(dataset) if dataset else default_dataset_path()
        with console.status("[bold cyan]Running retrieval eval...", spinner="dots"):
            result, json_path, md_path = evaluate(
                retriever,
                cfg,
                dataset_path=dataset_path,
                out_dir=Path(out),
                corpus_chunk_count=chunk_count,
                k_values=k_values,
                use_doc_type_filter=doc_type_filter,
                known_refs=known_refs,
                index_checker=lambda gold: check_index(
                    gold, store, ExactStore(store), embedder, cfg,
                    max_k=max(k_values), use_doc_type_filter=doc_type_filter,
                ),
            )
    except (EvalError, RAGError) as exc:
        console.print(Panel(f"[red]{exc}[/red]", title="Eval failed", border_style="red"))
        return 1

    console.print(render_table(result))
    if result.index_check is not None:
        console.print(index_check_line(result.index_check))
    console.print(f"[dim]JSON: {json_path}\nMarkdown: {md_path}[/dim]")
    return 0


def _cmd_eval_gen(per_doc_type: int, seed: int, out: str) -> int:
    """Draft gold-set candidates with the research LLM for human review.

    Args:
        per_doc_type: Chunks to sample per doc_type.
        seed: Sampling seed.
        out: Destination JSONL for the drafts; never overwritten.

    Returns:
        0 on success, 1 on a handled failure.
    """
    from pathlib import Path

    from quarq.eval.dataset import write_gold
    from quarq.eval.generate import draft_items, sample_chunks
    from quarq.exceptions import EvalError, RAGError
    from quarq.llm import get_llm
    from quarq.rag.store import VectorStore

    out_path = Path(out)
    cfg = load_config()
    try:
        if per_doc_type < 1:
            raise EvalError(f"--per-doc-type must be >= 1, got {per_doc_type}")
        if out_path.exists():
            raise EvalError(f"Refusing to overwrite {out_path}: choose a new --out path")
        store = VectorStore(cfg)
        if store.count() == 0:
            raise EvalError("RAG corpus is empty. Run: quarq rag add ./docs/")
        llm = get_llm(cfg, agent="research")
        chunks = sample_chunks(store.list_chunks(), per_doc_type=per_doc_type, seed=seed)
        with console.status(
            f"[bold cyan]Drafting {len(chunks)} questions...", spinner="dots"
        ):
            items, skipped = draft_items(chunks, llm)
        if not items:
            raise EvalError(
                f"No drafts produced ({skipped} chunks skipped): the LLM returned "
                "no usable questions. Nothing was written."
            )
        write_gold(items, out_path)
    except (EvalError, RAGError) as exc:
        console.print(Panel(f"[red]{exc}[/red]", title="eval-gen failed", border_style="red"))
        return 1

    console.print(
        Panel(
            f"Wrote {len(items)} drafts to [bold]{out_path}[/bold] ({skipped} skipped).\n"
            "These are NOT gold yet. For each row: open the PDF page, fix or delete the "
            "question, add any other correct pages, write a note, and set provenance to "
            "'synthetic-draft+human-accept'. Then copy accepted rows into "
            "quarq/eval/datasets/quarq_gold_v1.jsonl.\n"
            "Use the PDF viewer's page index (1 = first physical page), not the printed "
            "page number.",
            title="Drafts ready for review",
            border_style="green",
        )
    )
    return 0


def main() -> None:
    """Entry point for the quarq CLI."""
    parser = argparse.ArgumentParser(
        prog="quarq",
        description="French institutional portfolio analytics",
        add_help=True,
    )
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("status", help="Run checks and print status table")
    subparsers.add_parser("version", help="Print version and exit")

    # quarq query
    query_parser = subparsers.add_parser("query", help="Ask a question against the RAG corpus")
    query_parser.add_argument("question", help="Natural language question")
    query_parser.add_argument("--doc-type", default=None, help="Filter by doc_type (e.g. ecb_fsr)")
    query_parser.add_argument("--k", type=int, default=5, help="Number of chunks to retrieve")

    # quarq rag
    rag_parser = subparsers.add_parser("rag", help="Manage the RAG document corpus")
    rag_sub = rag_parser.add_subparsers(dest="rag_command")
    rag_sub.add_parser("status", help="Print corpus statistics")
    rag_add_parser = rag_sub.add_parser("add", help="Index a file or folder")
    rag_add_parser.add_argument("path", help="Path to a PDF file or folder")
    rag_manifest_parser = rag_sub.add_parser(
        "manifest", help="Apply the corpus manifest to chunks already indexed (no re-embedding)"
    )
    rag_manifest_parser.add_argument("path", help="Manifest file, or the folder containing it")

    # quarq config
    config_parser = subparsers.add_parser("config", help="View or update quarq configuration")
    config_parser.add_argument(
        "--set-lmstudio-url",
        metavar="URL",
        help="Set the LM Studio base URL, e.g. http://192.168.1.101:1234/v1",
    )

    # quarq serve
    serve_parser = subparsers.add_parser("serve", help="Start the FastAPI REST API server")
    serve_parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    serve_parser.add_argument("--port", type=int, default=8000, help="TCP port")
    serve_parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    # quarq report
    report_parser = subparsers.add_parser("report", help="Generate a portfolio report from a TOML file")
    report_parser.add_argument("--portfolio", required=True, help="Path to portfolio TOML file")
    report_parser.add_argument(
        "--format", default="html", choices=["html", "pdf", "json"], help="Output format"
    )
    report_parser.add_argument("--output", default=None, help="Output file path")
    report_parser.add_argument("--narrative", action="store_true", help="Include LLM narrative")
    report_parser.add_argument("--open", action="store_true", dest="open_browser",
                               help="Open report in browser after generation")

    # quarq eval
    eval_parser = subparsers.add_parser(
        "eval", help="Measure retrieval quality against a human-accepted gold set"
    )
    eval_parser.add_argument(
        "--dataset", default=None, help="Gold-set JSONL (default: packaged quarq_gold_v1)"
    )
    eval_parser.add_argument("--k", default="1,3,5", help="Comma-separated k values")
    eval_parser.add_argument("--out", default="reports", help="Directory for the reports")
    eval_parser.add_argument(
        "--doc-type-filter",
        action="store_true",
        help="Restrict each retrieval to the question's doc_type",
    )

    # quarq eval-gen
    eval_gen_parser = subparsers.add_parser(
        "eval-gen", help="Draft gold-set questions with the LLM for human review"
    )
    eval_gen_parser.add_argument(
        "--per-doc-type", type=int, default=20, help="Chunks to sample per doc_type"
    )
    eval_gen_parser.add_argument("--seed", type=int, default=7, help="Sampling seed")
    eval_gen_parser.add_argument(
        "--out", default="reports/eval_gen_drafts.jsonl", help="Draft JSONL (never overwritten)"
    )

    args = parser.parse_args()

    try:
        if args.command == "status":
            _cmd_status()
        elif args.command == "version":
            console.print(f"quarq {__version__}")
        elif args.command == "query":
            _cmd_query(args.question, args.doc_type, args.k)
        elif args.command == "rag":
            if args.rag_command == "status":
                _cmd_rag_status()
            elif args.rag_command == "add":
                _cmd_rag_add(args.path)
            elif args.rag_command == "manifest":
                code = _cmd_rag_manifest(args.path)
                if code:
                    sys.exit(code)
            else:
                rag_parser.print_help()
        elif args.command == "config":
            if args.set_lmstudio_url:
                _cmd_config_set_lmstudio_url(args.set_lmstudio_url)
            else:
                config_parser.print_help()
        elif args.command == "serve":
            _cmd_serve(args.host, args.port, args.reload)
        elif args.command == "report":
            _cmd_report(
                portfolio_path=args.portfolio,
                fmt=args.format,
                output=args.output,
                include_narrative=args.narrative,
                open_browser=args.open_browser,
            )
        elif args.command == "eval":
            code = _cmd_eval(args.dataset, args.k, args.out, args.doc_type_filter)
            if code:
                sys.exit(code)
        elif args.command == "eval-gen":
            code = _cmd_eval_gen(args.per_doc_type, args.seed, args.out)
            if code:
                sys.exit(code)
        else:
            _launch_loop()
    except KeyboardInterrupt:
        console.print("\n[dim]Goodbye.[/dim]")
        sys.exit(0)
