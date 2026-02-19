"""
FastAPI application factory.

We use the factory pattern (``create_app()``) instead of a module-level
``app`` object for two reasons:
  1. Tests can create isolated app instances with custom settings.
  2. It avoids import-time side effects (e.g. loading models, connecting
     to databases) that would run on every ``import code_indexer.api.app``.

Application lifecycle
----------------------
On startup (``lifespan``):
  * Build the embedder.
  * Build the vector store.
  * Mount the pipeline as an app-level state object.

On shutdown:
  * No explicit cleanup needed (connections are closed by garbage collection).

Dependency injection
---------------------
FastAPI's ``Depends`` mechanism is used to inject shared resources (pipeline,
embedder, vector store) into route handlers without global singletons.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from code_indexer.api.middleware import APIKeyMiddleware, RequestLoggingMiddleware
from code_indexer.api.routes import health, index, search
from code_indexer.core.config import get_settings
from code_indexer.indexer.pipeline import (
    IndexingPipeline,
    build_embedder,
    build_vector_store,
)

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application lifecycle.

    Runs once on startup and once on shutdown.  Shared resources are attached
    to ``app.state`` so route handlers can access them via ``request.app.state``.
    """
    settings = get_settings()

    logger.info(
        "Starting code-indexer",
        embedding_provider=settings.embedding.provider,
        vector_store_provider=settings.vectorstore.provider,
    )

    # Build shared components.
    embedder = build_embedder(settings)
    vector_store = build_vector_store(settings)
    pipeline = IndexingPipeline(
        settings=settings,
        embedder=embedder,
        vector_store=vector_store,
    )

    # Attach to app state for dependency injection.
    app.state.settings = settings
    app.state.pipeline = pipeline
    app.state.embedder = embedder
    app.state.vector_store = vector_store

    logger.info("code-indexer started successfully")
    yield  # hand control to the application

    logger.info("code-indexer shutting down")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        A fully configured ``FastAPI`` instance ready to serve.
    """
    settings = get_settings()

    app = FastAPI(
        title="Code Indexer API",
        description=(
            "Production-ready codebase indexing system for RAG and code generation. "
            "Index your codebase with Tree-sitter and query it with natural language."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ---------------------------------------------------------------------------
    # Middleware (applied in reverse order — last added runs first)
    # ---------------------------------------------------------------------------

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Structured request logging.
    app.add_middleware(RequestLoggingMiddleware)

    # Optional API key authentication.
    if settings.api.api_key:
        app.add_middleware(APIKeyMiddleware, api_key=settings.api.api_key)

    # ---------------------------------------------------------------------------
    # Routers
    # ---------------------------------------------------------------------------

    app.include_router(health.router, tags=["health"])
    app.include_router(index.router, prefix="/index", tags=["indexing"])
    app.include_router(search.router, prefix="/search", tags=["search"])

    # ---------------------------------------------------------------------------
    # Global exception handlers
    # ---------------------------------------------------------------------------

    from fastapi import Request  # noqa: PLC0415
    from fastapi.responses import JSONResponse  # noqa: PLC0415

    from code_indexer.core.exceptions import CodeIndexerError  # noqa: PLC0415

    @app.exception_handler(CodeIndexerError)
    async def code_indexer_error_handler(
        request: Request, exc: CodeIndexerError
    ) -> JSONResponse:
        status_code = getattr(exc, "http_status", 500)
        return JSONResponse(
            status_code=status_code,
            content={"error": type(exc).__name__, "detail": str(exc)},
        )

    return app


# Module-level ``app`` for uvicorn to discover when running:
#   uvicorn code_indexer.api.app:app
app = create_app()
