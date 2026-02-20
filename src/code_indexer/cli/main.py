"""
Command-line interface for code-indexer.

Usage
------
    code-indexer index ./my-project
    code-indexer search "parse JWT token"
    code-indexer stats
    code-indexer serve

All commands respect the same environment variables / .env file as the
API server.  You can override settings per-command with flags.

Rich is used for formatted terminal output (tables, progress bars, syntax
highlighted code snippets).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.syntax import Syntax
from rich.table import Table

console = Console()


# ---------------------------------------------------------------------------
# Root group
# ---------------------------------------------------------------------------


@click.group()
@click.version_option(version="1.0.0")
def cli() -> None:
    """Code Indexer – index and search codebases for RAG / code gen."""


# ---------------------------------------------------------------------------
# index command
# ---------------------------------------------------------------------------


@cli.command("index")
@click.argument("path", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option("--strategy", default=None, help="Chunking strategy (ast/token/sliding_window).")
@click.option("--clear", is_flag=True, default=False, help="Wipe existing index first.")
@click.option("--provider", default=None, help="Embedding provider override.")
@click.option("--store", default=None, help="Vector store provider override.")
@click.option(
    "--metadata",
    multiple=True,
    help="Extra metadata as key=value pairs (repeatable).",
)
def index_command(
    path: str,
    strategy: str | None,
    clear: bool,
    provider: str | None,
    store: str | None,
    metadata: tuple[str, ...],
) -> None:
    """Index a codebase directory.

    PATH is the root directory of the codebase to index.

    Examples::

        code-indexer index ./my-project
        code-indexer index ./my-project --strategy token --clear
        code-indexer index ./my-project --metadata branch=main --metadata author=alice
    """
    from code_indexer.core.config import get_settings  # noqa: PLC0415
    from code_indexer.indexer.pipeline import (  # noqa: PLC0415
        IndexingPipeline,
        build_chunker,
        build_embedder,
        build_vector_store,
    )

    settings = get_settings()

    # Apply CLI overrides.
    if strategy:
        settings.chunker.strategy = strategy  # type: ignore[assignment]
    if provider:
        settings.embedding.provider = provider  # type: ignore[assignment]
    if store:
        settings.vectorstore.provider = store  # type: ignore[assignment]

    # Parse extra metadata.
    extra_metadata: dict = {}
    for item in metadata:
        if "=" in item:
            k, _, v = item.partition("=")
            extra_metadata[k.strip()] = v.strip()

    # Progress bar integration.
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    )

    task_id = None

    def progress_cb(message: str, current: int, total: int) -> None:
        nonlocal task_id
        if task_id is None and total > 0:
            task_id = progress.add_task(message, total=total)
        if task_id is not None:
            progress.update(task_id, description=message, completed=current, total=max(total, 1))

    console.print(Panel(f"[bold]Indexing:[/bold] {path}", style="cyan"))

    with progress:
        pipeline = IndexingPipeline(
            settings=settings,
            progress_cb=progress_cb,
        )
        result = pipeline.index_directory(
            path,
            extra_metadata=extra_metadata,
            clear_existing=clear,
        )

    # Summary table.
    table = Table(title="Indexing Summary", show_header=True)
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Files processed", str(result.files_processed))
    table.add_row("Files skipped", str(result.files_skipped))
    table.add_row("Chunks produced", str(result.chunks_produced))
    table.add_row("Chunks embedded", str(result.chunks_embedded))
    table.add_row("Chunks stored", str(result.chunks_stored))
    table.add_row("Elapsed", f"{result.elapsed_seconds:.1f}s")
    table.add_row("Errors", str(len(result.errors)))

    console.print(table)

    if result.errors:
        console.print("\n[bold red]Errors:[/bold red]")
        for file_path, error in result.errors[:10]:
            console.print(f"  [red]{file_path}[/red]: {error}")


# ---------------------------------------------------------------------------
# search command
# ---------------------------------------------------------------------------


@cli.command("search")
@click.argument("query")
@click.option("--top-k", default=5, show_default=True, help="Number of results.")
@click.option("--language", default=None, help="Filter by language (python/javascript/…).")
@click.option("--type", "chunk_type", default=None, help="Filter by chunk type (function/class/…).")
@click.option("--min-score", default=0.0, show_default=True, help="Minimum similarity score.")
@click.option("--context", is_flag=True, default=False, help="Include surrounding context.")
@click.option("--json-output", is_flag=True, default=False, help="Output raw JSON.")
def search_command(
    query: str,
    top_k: int,
    language: str | None,
    chunk_type: str | None,
    min_score: float,
    context: bool,
    json_output: bool,
) -> None:
    """Search the indexed codebase.

    QUERY is a natural language or code query string.

    Examples::

        code-indexer search "validate JWT token"
        code-indexer search "class UserService" --language python --type class
        code-indexer search "parse config" --top-k 10 --min-score 0.4 --context
    """
    from code_indexer.core.config import get_settings  # noqa: PLC0415
    from code_indexer.core.models import ChunkType, Language, SearchQuery  # noqa: PLC0415
    from code_indexer.indexer.pipeline import build_embedder, build_vector_store  # noqa: PLC0415
    from code_indexer.retrieval.retriever import CodeRetriever  # noqa: PLC0415

    settings = get_settings()
    embedder = build_embedder(settings)
    vector_store = build_vector_store(settings)

    if embedder is None:
        console.print("[red]Embedder not configured.[/red]")
        sys.exit(1)
    if vector_store is None:
        console.print("[red]Vector store not configured.[/red]")
        sys.exit(1)

    lang = Language(language) if language else None
    c_type = [ChunkType(chunk_type)] if chunk_type else None

    search_query = SearchQuery(
        query=query,
        top_k=top_k,
        language=lang,
        chunk_types=c_type,
        min_score=min_score,
        include_context=context,
    )

    with console.status("Searching …"):
        retriever = CodeRetriever(embedder=embedder, vector_store=vector_store)
        response = retriever.search(search_query)

    if json_output:
        click.echo(response.model_dump_json(indent=2))
        return

    console.print(
        f"\n[bold]Query:[/bold] {query}  "
        f"[dim]({response.total} results, {response.latency_ms:.0f}ms)[/dim]\n"
    )

    for result in response.results:
        chunk = result.chunk
        header = (
            f"[bold cyan]#{result.rank}[/bold cyan]  "
            f"[green]{chunk.path}[/green]  "
            f"[yellow]{chunk.language.value}[/yellow]  "
            f"[magenta]{chunk.chunk_type.value}[/magenta]"
        )
        if chunk.name:
            header += f" [bold]{chunk.name}[/bold]"
        header += f"  [dim]score={result.score:.3f}  lines {chunk.start_line}–{chunk.end_line}[/dim]"

        console.print(header)

        if context and chunk.context_before:
            console.print(
                Syntax(
                    chunk.context_before,
                    chunk.language.value,
                    theme="monokai",
                    line_numbers=False,
                ),
                style="dim",
            )

        console.print(
            Syntax(
                chunk.content,
                chunk.language.value,
                theme="monokai",
                line_numbers=True,
                start_line=chunk.start_line,
            )
        )

        if context and chunk.context_after:
            console.print(
                Syntax(
                    chunk.context_after,
                    chunk.language.value,
                    theme="monokai",
                    line_numbers=False,
                ),
                style="dim",
            )

        console.print()


# ---------------------------------------------------------------------------
# stats command
# ---------------------------------------------------------------------------


@cli.command("stats")
@click.option("--json-output", is_flag=True, default=False, help="Output raw JSON.")
def stats_command(json_output: bool) -> None:
    """Show index statistics."""
    from code_indexer.core.config import get_settings  # noqa: PLC0415
    from code_indexer.indexer.pipeline import build_vector_store  # noqa: PLC0415

    settings = get_settings()
    vector_store = build_vector_store(settings)

    if vector_store is None:
        console.print("[red]Vector store not configured.[/red]")
        sys.exit(1)

    stats = vector_store.get_stats()

    if json_output:
        click.echo(stats.model_dump_json(indent=2))
        return

    table = Table(title="Index Statistics", show_header=True)
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Total files", str(stats.total_files))
    table.add_row("Total chunks", str(stats.total_chunks))
    table.add_row("Total tokens", f"{stats.total_tokens:,}")
    table.add_row("Vector store", stats.vector_store)
    table.add_row("Embedding model", stats.embedding_model)

    console.print(table)

    if stats.languages:
        lang_table = Table(title="Languages", show_header=True)
        lang_table.add_column("Language")
        lang_table.add_column("Chunks", justify="right")
        for lang, count in sorted(stats.languages.items(), key=lambda x: -x[1]):
            lang_table.add_row(lang, str(count))
        console.print(lang_table)


# ---------------------------------------------------------------------------
# serve command
# ---------------------------------------------------------------------------


@cli.command("serve")
@click.option("--host", default=None, help="Bind host.")
@click.option("--port", default=None, type=int, help="Bind port.")
@click.option("--workers", default=None, type=int, help="Number of workers.")
@click.option("--reload", is_flag=True, default=False, help="Enable hot reload (dev mode).")
def serve_command(
    host: str | None,
    port: int | None,
    workers: int | None,
    reload: bool,
) -> None:
    """Start the API server.

    Example::

        code-indexer serve --port 8080
        code-indexer serve --host 0.0.0.0 --workers 4
    """
    import uvicorn  # noqa: PLC0415

    from code_indexer.core.config import get_settings  # noqa: PLC0415

    settings = get_settings()
    uvicorn.run(
        "code_indexer.api.app:app",
        host=host or settings.api.host,
        port=port or settings.api.port,
        workers=workers or settings.api.workers,
        reload=reload or settings.api.reload,
        log_level=settings.api.log_level,
    )


# ---------------------------------------------------------------------------
# graph commands
# ---------------------------------------------------------------------------


@cli.group("graph")
def graph_group() -> None:
    """Code knowledge graph commands (structural relationships)."""


@graph_group.command("index")
@click.argument("path", type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option("--clear", is_flag=True, default=False, help="Wipe existing graph first.")
@click.option("--provider", default=None, help="Graph store provider (in_memory/neo4j).")
def graph_index_command(path: str, clear: bool, provider: str | None) -> None:
    """Index the code knowledge graph for a directory.

    This builds the structural graph (symbols, imports, call graph, inheritance)
    separately from the vector index.

    Examples::

        code-indexer graph index ./my-project
        code-indexer graph index ./my-project --clear --provider neo4j
    """
    from code_indexer.core.config import get_settings  # noqa: PLC0415
    from code_indexer.graph.pipeline import GraphIndexingPipeline, build_graph_store  # noqa: PLC0415

    settings = get_settings()
    if provider:
        settings.graph.provider = provider  # type: ignore[assignment]

    console.print(Panel(f"[bold]Graph Indexing:[/bold] {path}", style="green"))

    with console.status("Building code knowledge graph …"):
        graph_store = build_graph_store(settings)
        pipeline = GraphIndexingPipeline(settings=settings, graph_store=graph_store)
        result = pipeline.index_directory(path, clear_existing=clear)

    table = Table(title="Graph Indexing Summary", show_header=True)
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("Files processed", str(result.files_processed))
    table.add_row("Files skipped", str(result.files_skipped))
    table.add_row("Symbols created", str(result.symbols_created))
    table.add_row("Edges created", str(result.edges_created))
    table.add_row("Elapsed", f"{result.elapsed_seconds:.1f}s")
    table.add_row("Errors", str(len(result.errors)))
    console.print(table)


@graph_group.command("callers")
@click.argument("symbol_name")
@click.option("--depth", default=1, show_default=True, help="Traversal depth.")
@click.option("--json-output", is_flag=True, default=False)
def graph_callers_command(symbol_name: str, depth: int, json_output: bool) -> None:
    """Show what calls SYMBOL_NAME in the codebase."""
    from code_indexer.core.config import get_settings  # noqa: PLC0415
    from code_indexer.graph.pipeline import GraphIndexingPipeline, build_graph_store  # noqa: PLC0415

    settings = get_settings()
    store = build_graph_store(settings)
    syms = store.find_symbols_by_name(symbol_name)

    if not syms:
        console.print(f"[red]Symbol {symbol_name!r} not found in graph.[/red]")
        sys.exit(1)

    callers = store.get_callers(syms[0].id, depth=depth)

    if json_output:
        click.echo(json.dumps([c.model_dump() for c in callers], indent=2))
        return

    console.print(f"\n[bold]Callers of '{symbol_name}':[/bold]\n")
    for c in callers:
        console.print(f"  [green]{c.file_path}[/green]:{c.start_line}  [cyan]{c.qualified_name}[/cyan]  [{c.kind.value}]")


@graph_group.command("callees")
@click.argument("symbol_name")
@click.option("--depth", default=1, show_default=True, help="Traversal depth.")
@click.option("--json-output", is_flag=True, default=False)
def graph_callees_command(symbol_name: str, depth: int, json_output: bool) -> None:
    """Show what SYMBOL_NAME calls in the codebase."""
    from code_indexer.core.config import get_settings  # noqa: PLC0415
    from code_indexer.graph.pipeline import GraphIndexingPipeline, build_graph_store  # noqa: PLC0415

    settings = get_settings()
    store = build_graph_store(settings)
    syms = store.find_symbols_by_name(symbol_name)

    if not syms:
        console.print(f"[red]Symbol {symbol_name!r} not found in graph.[/red]")
        sys.exit(1)

    callees = store.get_callees(syms[0].id, depth=depth)

    if json_output:
        click.echo(json.dumps([c.model_dump() for c in callees], indent=2))
        return

    console.print(f"\n[bold]'{symbol_name}' calls:[/bold]\n")
    for c in callees:
        console.print(f"  [green]{c.file_path}[/green]:{c.start_line}  [cyan]{c.qualified_name}[/cyan]  [{c.kind.value}]")


@graph_group.command("context")
@click.argument("symbol_name")
@click.option("--json-output", is_flag=True, default=False)
def graph_context_command(symbol_name: str, json_output: bool) -> None:
    """Show full structural context for SYMBOL_NAME (for RAG).

    Prints callers, callees, parent class, inheritance chain, and imports.
    """
    from code_indexer.core.config import get_settings  # noqa: PLC0415
    from code_indexer.graph.pipeline import build_graph_store  # noqa: PLC0415

    settings = get_settings()
    store = build_graph_store(settings)
    syms = store.find_symbols_by_name(symbol_name)

    if not syms:
        console.print(f"[red]Symbol {symbol_name!r} not found in graph.[/red]")
        sys.exit(1)

    ctx = store.get_symbol_context(syms[0].id)
    if ctx is None:
        console.print(f"[red]No context found for {symbol_name!r}.[/red]")
        sys.exit(1)

    if json_output:
        click.echo(ctx.model_dump_json(indent=2))
        return

    console.print(Panel(ctx.to_context_text(), title=f"Context: {symbol_name}", style="cyan"))


@graph_group.command("stats")
@click.option("--json-output", is_flag=True, default=False)
def graph_stats_command(json_output: bool) -> None:
    """Show code knowledge graph statistics."""
    from code_indexer.core.config import get_settings  # noqa: PLC0415
    from code_indexer.graph.pipeline import build_graph_store  # noqa: PLC0415

    settings = get_settings()
    store = build_graph_store(settings)
    stats = store.get_stats()

    if json_output:
        click.echo(stats.model_dump_json(indent=2))
        return

    table = Table(title="Graph Statistics", show_header=True)
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_row("Files", str(stats.total_files))
    table.add_row("Symbols", str(stats.total_symbols))
    table.add_row("Directories", str(stats.total_directories))
    table.add_row("Edges", str(stats.total_edges))
    table.add_row("Graph store", stats.graph_store)
    console.print(table)

    if stats.edges_by_type:
        edge_table = Table(title="Edges by Type")
        edge_table.add_column("Type")
        edge_table.add_column("Count", justify="right")
        for rel_type, count in sorted(stats.edges_by_type.items(), key=lambda x: -x[1]):
            edge_table.add_row(rel_type, str(count))
        console.print(edge_table)

    if stats.symbols_by_kind:
        kind_table = Table(title="Symbols by Kind")
        kind_table.add_column("Kind")
        kind_table.add_column("Count", justify="right")
        for kind, count in sorted(stats.symbols_by_kind.items(), key=lambda x: -x[1]):
            kind_table.add_row(kind, str(count))
        console.print(kind_table)


if __name__ == "__main__":
    cli()
