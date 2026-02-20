"""
Custom exception hierarchy for code-indexer.

Having a clear hierarchy lets callers catch at the right level of granularity
without importing individual exception classes::

    try:
        result = indexer.index(path)
    except IndexerError as exc:          # catch anything from the indexer
        logger.error("indexing failed", error=str(exc))
    except CodeIndexerError:             # catch anything from the whole lib
        ...

HTTP status codes are attached to API-layer exceptions so that FastAPI
exception handlers can map them directly to HTTP responses.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


class CodeIndexerError(Exception):
    """Base exception for all code-indexer errors."""

    def __init__(self, message: str, *, detail: str | None = None) -> None:
        super().__init__(message)
        self.detail = detail or message


# ---------------------------------------------------------------------------
# Parser errors
# ---------------------------------------------------------------------------


class ParserError(CodeIndexerError):
    """Raised when the Tree-sitter parser layer fails."""


class UnsupportedLanguageError(ParserError):
    """Raised when a file's language is not supported."""

    def __init__(self, language: str) -> None:
        super().__init__(f"Unsupported language: {language!r}")
        self.language = language


class FileTooLargeError(ParserError):
    """Raised when a file exceeds the configured size limit."""

    def __init__(self, path: str, size_bytes: int, limit_bytes: int) -> None:
        super().__init__(
            f"File {path!r} is {size_bytes:,} bytes, limit is {limit_bytes:,} bytes"
        )
        self.path = path
        self.size_bytes = size_bytes
        self.limit_bytes = limit_bytes


# ---------------------------------------------------------------------------
# Chunker errors
# ---------------------------------------------------------------------------


class ChunkerError(CodeIndexerError):
    """Raised when a chunking strategy fails."""


class ChunkTooSmallError(ChunkerError):
    """Raised when a chunk is smaller than the configured minimum."""


# ---------------------------------------------------------------------------
# Embedding errors
# ---------------------------------------------------------------------------


class EmbeddingError(CodeIndexerError):
    """Raised when an embedding backend fails."""


class EmbeddingProviderError(EmbeddingError):
    """Raised when the embedding provider returns an unexpected error."""

    http_status: int = 502


class EmbeddingRateLimitError(EmbeddingError):
    """Raised when the provider rate-limits the request."""

    http_status: int = 429


# ---------------------------------------------------------------------------
# Vector store errors
# ---------------------------------------------------------------------------


class VectorStoreError(CodeIndexerError):
    """Raised when a vector store operation fails."""


class CollectionNotFoundError(VectorStoreError):
    """Raised when a named collection does not exist in the vector store."""

    http_status: int = 404

    def __init__(self, collection: str) -> None:
        super().__init__(f"Collection not found: {collection!r}")
        self.collection = collection


# ---------------------------------------------------------------------------
# Indexer errors
# ---------------------------------------------------------------------------


class IndexerError(CodeIndexerError):
    """Raised when the high-level indexing pipeline fails."""


class PathNotFoundError(IndexerError):
    """Raised when the supplied codebase path does not exist."""

    http_status: int = 404

    def __init__(self, path: str) -> None:
        super().__init__(f"Path not found: {path!r}")
        self.path = path


# ---------------------------------------------------------------------------
# Graph errors
# ---------------------------------------------------------------------------


class GraphError(CodeIndexerError):
    """Raised when a graph store operation fails."""


class GraphStoreConnectionError(GraphError):
    """Raised when the graph store cannot be reached."""

    http_status: int = 503


class SymbolNotFoundError(GraphError):
    """Raised when a requested symbol does not exist in the graph."""

    http_status: int = 404

    def __init__(self, symbol_id: str) -> None:
        super().__init__(f"Symbol not found: {symbol_id!r}")
        self.symbol_id = symbol_id


# ---------------------------------------------------------------------------
# API / auth errors
# ---------------------------------------------------------------------------


class AuthenticationError(CodeIndexerError):
    """Raised when an API request fails authentication."""

    http_status: int = 401


class AuthorizationError(CodeIndexerError):
    """Raised when an API request is not authorised."""

    http_status: int = 403
