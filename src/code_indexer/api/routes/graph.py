"""Graph traversal REST API endpoints.

All endpoints operate on the code knowledge graph populated by
:class:`~code_indexer.graph.pipeline.GraphIndexingPipeline`.

Endpoints
---------
POST  /graph/index            — Index codebase graph (separate from vector index)
GET   /graph/stats            — Graph statistics
DELETE /graph/clear           — Wipe graph
GET   /graph/symbol/{id}      — Symbol detail + full context
POST  /graph/symbol/search    — Find symbols by name
GET   /graph/file/{file_id}   — All symbols in a file
POST  /graph/callers          — Who calls a symbol?
POST  /graph/callees          — What does a symbol call?
POST  /graph/call-path        — Shortest call path between two symbols
POST  /graph/import-graph     — Import graph for a file
POST  /graph/subclasses       — Subclasses of a symbol
POST  /graph/superclasses     — Base classes of a symbol
POST  /graph/context          — Full structural context (for RAG)
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from code_indexer.graph.models import GraphStats, SymbolContext, SymbolNode
from code_indexer.graph.pipeline import GraphIndexResult, GraphIndexingPipeline

logger = logging.getLogger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class GraphIndexRequest(BaseModel):
    path: str
    clear_existing: bool = False


class GraphIndexResponse(BaseModel):
    files_processed: int
    files_skipped: int
    symbols_created: int
    edges_created: int
    elapsed_seconds: float
    errors: list[list[str]] = []


class SymbolSearchRequest(BaseModel):
    name: str
    kind: str | None = None
    file_path: str | None = None


class CallersRequest(BaseModel):
    symbol_id: str
    depth: int = 1


class CalleesRequest(BaseModel):
    symbol_id: str
    depth: int = 1


class CallPathRequest(BaseModel):
    from_symbol_id: str
    to_symbol_id: str


class ImportGraphRequest(BaseModel):
    file_id: str


class SubclassesRequest(BaseModel):
    symbol_id: str


class SuperclassesRequest(BaseModel):
    symbol_id: str


class ContextRequest(BaseModel):
    symbol_id: str | None = None
    symbol_name: str | None = None
    """If ``symbol_id`` is not provided, the first symbol with this name is used."""


# ---------------------------------------------------------------------------
# Dependency helper
# ---------------------------------------------------------------------------


def _graph_pipeline(request: Request) -> GraphIndexingPipeline:
    pipeline = getattr(request.app.state, "graph_pipeline", None)
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Graph pipeline is not initialised. "
                "Start the server with GRAPH_ENABLED=true or POST /graph/index first."
            ),
        )
    return pipeline


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------


@router.post("/index", response_model=GraphIndexResponse)
async def graph_index(req: GraphIndexRequest, request: Request) -> GraphIndexResponse:
    """Index the code knowledge graph for a directory."""
    pipeline = _graph_pipeline(request)
    loop = asyncio.get_event_loop()
    result: GraphIndexResult = await loop.run_in_executor(
        None,
        lambda: pipeline.index_directory(
            req.path, clear_existing=req.clear_existing
        ),
    )
    return GraphIndexResponse(
        files_processed=result.files_processed,
        files_skipped=result.files_skipped,
        symbols_created=result.symbols_created,
        edges_created=result.edges_created,
        elapsed_seconds=result.elapsed_seconds,
        errors=[[p, e] for p, e in result.errors],
    )


# ---------------------------------------------------------------------------
# Stats & management
# ---------------------------------------------------------------------------


@router.get("/stats", response_model=GraphStats)
async def graph_stats(request: Request) -> GraphStats:
    """Return code knowledge graph statistics."""
    return _graph_pipeline(request).get_stats()


@router.delete("/clear")
async def graph_clear(request: Request) -> dict[str, str]:
    """Wipe the entire graph index."""
    _graph_pipeline(request).store.clear()
    return {"status": "cleared"}


# ---------------------------------------------------------------------------
# Symbol lookup
# ---------------------------------------------------------------------------


@router.get("/symbol/{symbol_id}", response_model=SymbolNode | None)
async def get_symbol(symbol_id: str, request: Request) -> SymbolNode | None:
    """Get a symbol by its ID."""
    sym = _graph_pipeline(request).store.get_symbol_by_id(symbol_id)
    if sym is None:
        raise HTTPException(status_code=404, detail=f"Symbol {symbol_id!r} not found")
    return sym


@router.post("/symbol/search", response_model=list[SymbolNode])
async def search_symbols(req: SymbolSearchRequest, request: Request) -> list[SymbolNode]:
    """Find symbols by name, with optional kind and file_path filters."""
    return _graph_pipeline(request).store.find_symbols_by_name(
        req.name, kind=req.kind, file_path=req.file_path
    )


@router.get("/file/{file_id}/symbols", response_model=list[SymbolNode])
async def symbols_in_file(file_id: str, request: Request) -> list[SymbolNode]:
    """Return all symbols defined in a file."""
    return _graph_pipeline(request).store.find_symbols_in_file(file_id)


# ---------------------------------------------------------------------------
# Graph traversal
# ---------------------------------------------------------------------------


@router.post("/callers", response_model=list[SymbolNode])
async def get_callers(req: CallersRequest, request: Request) -> list[SymbolNode]:
    """Return all symbols that call the given symbol (up to ``depth`` hops)."""
    return _graph_pipeline(request).store.get_callers(req.symbol_id, depth=req.depth)


@router.post("/callees", response_model=list[SymbolNode])
async def get_callees(req: CalleesRequest, request: Request) -> list[SymbolNode]:
    """Return all symbols called by the given symbol (up to ``depth`` hops)."""
    return _graph_pipeline(request).store.get_callees(req.symbol_id, depth=req.depth)


@router.post("/call-path", response_model=list[SymbolNode] | None)
async def get_call_path(req: CallPathRequest, request: Request) -> list[SymbolNode] | None:
    """Find the shortest call path between two symbols."""
    return _graph_pipeline(request).store.get_call_path(
        req.from_symbol_id, req.to_symbol_id
    )


@router.post("/import-graph", response_model=dict[str, list[str]])
async def get_import_graph(req: ImportGraphRequest, request: Request) -> dict[str, list[str]]:
    """Return the import adjacency map for files reachable from the given file."""
    return _graph_pipeline(request).store.get_import_graph(req.file_id)


@router.post("/subclasses", response_model=list[SymbolNode])
async def get_subclasses(req: SubclassesRequest, request: Request) -> list[SymbolNode]:
    """Return all direct subclasses / implementors of the given symbol."""
    return _graph_pipeline(request).store.get_subclasses(req.symbol_id)


@router.post("/superclasses", response_model=list[SymbolNode])
async def get_superclasses(req: SuperclassesRequest, request: Request) -> list[SymbolNode]:
    """Return all direct base classes / interfaces of the given symbol."""
    return _graph_pipeline(request).store.get_superclasses(req.symbol_id)


@router.post("/context", response_model=SymbolContext | None)
async def get_context(req: ContextRequest, request: Request) -> SymbolContext | None:
    """Return full structural context for a symbol (the main RAG query).

    Includes: callers, callees, parent class, methods (if class), inheritance
    chain, and file-level imports.
    """
    store = _graph_pipeline(request).store
    symbol_id = req.symbol_id

    if symbol_id is None and req.symbol_name:
        symbols = store.find_symbols_by_name(req.symbol_name)
        if not symbols:
            raise HTTPException(
                status_code=404,
                detail=f"No symbol named {req.symbol_name!r} found in the graph.",
            )
        symbol_id = symbols[0].id

    if symbol_id is None:
        raise HTTPException(status_code=422, detail="Provide either symbol_id or symbol_name.")

    ctx = store.get_symbol_context(symbol_id)
    if ctx is None:
        raise HTTPException(status_code=404, detail=f"Symbol {symbol_id!r} not found.")
    return ctx
