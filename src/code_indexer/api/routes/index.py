"""
Indexing API routes.

Endpoints:
  POST /index/directory   – Index an entire directory.
  POST /index/file        – Index a single file.
  DELETE /index/file      – Remove a file's chunks from the index.
  GET  /index/stats       – Return index statistics.
  DELETE /index/clear     – Wipe the entire index.

All heavy operations are synchronous internally but FastAPI runs them in
a thread pool (via ``run_in_executor``) so the event loop remains free.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from code_indexer.core.models import IndexStats

router = APIRouter()


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class IndexDirectoryRequest(BaseModel):
    """Request body for POST /index/directory."""

    path: str = Field(description="Absolute path to the codebase root directory.")
    clear_existing: bool = Field(
        default=False,
        description="Wipe existing index before re-indexing.  Use for clean re-index runs.",
    )
    extra_metadata: dict = Field(
        default_factory=dict,
        description="Arbitrary key-value metadata attached to every indexed chunk.",
    )


class IndexFileRequest(BaseModel):
    """Request body for POST /index/file."""

    path: str = Field(description="Absolute path to the source file.")


class IndexDirectoryResponse(BaseModel):
    """Response for POST /index/directory."""

    files_processed: int
    files_skipped: int
    chunks_produced: int
    chunks_embedded: int
    chunks_stored: int
    elapsed_seconds: float
    errors: list[dict]


class DeleteFileRequest(BaseModel):
    """Request body for DELETE /index/file."""

    source_file_id: str = Field(description="ID of the source file to remove.")


# ---------------------------------------------------------------------------
# Route handlers
# ---------------------------------------------------------------------------


@router.post("/directory", response_model=IndexDirectoryResponse)
async def index_directory(body: IndexDirectoryRequest, request: Request) -> IndexDirectoryResponse:
    """Index all source files in a directory tree.

    This is a blocking operation that can take minutes for large codebases.
    Consider running it as a background task in production.
    """
    pipeline = request.app.state.pipeline

    if not Path(body.path).is_dir():
        raise HTTPException(
            status_code=404,
            detail=f"Directory not found: {body.path!r}",
        )

    # Run the blocking pipeline in a thread pool to avoid blocking the event loop.
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: pipeline.index_directory(
            body.path,
            extra_metadata=body.extra_metadata,
            clear_existing=body.clear_existing,
        ),
    )

    return IndexDirectoryResponse(
        files_processed=result.files_processed,
        files_skipped=result.files_skipped,
        chunks_produced=result.chunks_produced,
        chunks_embedded=result.chunks_embedded,
        chunks_stored=result.chunks_stored,
        elapsed_seconds=result.elapsed_seconds,
        errors=[{"path": p, "error": e} for p, e in result.errors],
    )


@router.post("/file", response_model=IndexDirectoryResponse)
async def index_file(body: IndexFileRequest, request: Request) -> IndexDirectoryResponse:
    """Index a single source file.

    Useful for incremental updates when only one file has changed.
    """
    pipeline = request.app.state.pipeline

    if not Path(body.path).is_file():
        raise HTTPException(
            status_code=404,
            detail=f"File not found: {body.path!r}",
        )

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, lambda: pipeline.index_file(body.path))

    return IndexDirectoryResponse(
        files_processed=result.files_processed,
        files_skipped=result.files_skipped,
        chunks_produced=result.chunks_produced,
        chunks_embedded=result.chunks_embedded,
        chunks_stored=result.chunks_stored,
        elapsed_seconds=result.elapsed_seconds,
        errors=[{"path": p, "error": e} for p, e in result.errors],
    )


@router.get("/stats", response_model=IndexStats)
async def get_stats(request: Request) -> IndexStats:
    """Return current index statistics."""
    pipeline = request.app.state.pipeline
    stats = pipeline.get_stats()
    if stats is None:
        raise HTTPException(status_code=503, detail="Vector store not configured")
    return stats


@router.delete("/file")
async def delete_file(body: DeleteFileRequest, request: Request) -> dict:
    """Remove all chunks for a source file from the index."""
    vector_store = request.app.state.vector_store
    if vector_store is None:
        raise HTTPException(status_code=503, detail="Vector store not configured")

    deleted = vector_store.delete_by_file(body.source_file_id)
    return {"deleted_chunks": deleted, "source_file_id": body.source_file_id}


@router.delete("/clear")
async def clear_index(request: Request) -> dict:
    """Wipe the entire index.  This is irreversible."""
    vector_store = request.app.state.vector_store
    if vector_store is None:
        raise HTTPException(status_code=503, detail="Vector store not configured")
    vector_store.clear()
    return {"status": "cleared"}
