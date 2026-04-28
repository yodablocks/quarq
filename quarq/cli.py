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
    table.add_row("Collection", "quarq_rag_v1")
    table.add_row("Chroma path", cfg.rag.chroma_path)
    console.print(table)


def _cmd_rag_add(path_str: str) -> None:
    """Index a file or folder into the RAG corpus.

    Args:
        path_str: Path to a PDF file or folder of PDFs.
    """
    from rich.progress import Progress, SpinnerColumn, TextColumn

    from quarq.rag.embedder import Embedder
    from quarq.rag.loader import load_folder, load_pdf
    from quarq.rag.store import VectorStore

    cfg = load_config()
    target = Path(path_str).expanduser().resolve()

    if not target.exists():
        console.print(f"[red]Path does not exist: {target}[/red]")
        sys.exit(1)

    with console.status("[bold cyan]Loading documents...", spinner="dots"):
        if target.is_file():
            documents = load_pdf(target, chunk_size=cfg.rag.chunk_size,
                                 chunk_overlap=cfg.rag.chunk_overlap)
        else:
            documents = load_folder(target, chunk_size=cfg.rag.chunk_size,
                                    chunk_overlap=cfg.rag.chunk_overlap)

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
            else:
                rag_parser.print_help()
        else:
            _launch_loop()
    except KeyboardInterrupt:
        console.print("\n[dim]Goodbye.[/dim]")
        sys.exit(0)
