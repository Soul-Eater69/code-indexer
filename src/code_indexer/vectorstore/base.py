"""
Abstract base class for vector store backends.

A vector store is responsible for:
1. Storing ``EmbeddedChunk`` objects (persist vectors + metadata).
2. Retrieving the *k* most similar chunks to a query vector.
3. Deleting chunks by source file ID (needed for re-indexing).

All vector stores share the same interface so the retrieval layer is
completely backend-agnostic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from code_indexer.core.models import EmbeddedChunk, IndexStats, SearchResult


class BaseVectorStore(ABC):
    """Abstract vector store interface.

    Args:
        collection_name: Logical namespace for this index.
        dimensions:      Embedding vector dimensionality.
        distance_metric: Similarity function (``"cosine"``, ``"dot"``,
                         ``"euclidean"``).
    """

    def __init__(
        self,
        collection_name: str = "code_index",
        dimensions: int = 384,
        distance_metric: str = "cosine",
    ) -> None:
        self.collection_name = collection_name
        self.dimensions = dimensions
        self.distance_metric = distance_metric

    @abstractmethod
    def upsert(self, embedded_chunks: list[EmbeddedChunk]) -> int:
        """Insert or update *embedded_chunks* in the store.

        ``upsert`` semantics: if a chunk with the same ``chunk.id`` already
        exists it is replaced.  This makes re-indexing idempotent.

        Args:
            embedded_chunks: Chunks with their embedding vectors.

        Returns:
            Number of chunks successfully stored.
        """

    @abstractmethod
    def query(
        self,
        query_vector: list[float],
        top_k: int = 10,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        """Return the *top_k* most similar chunks to *query_vector*.

        Args:
            query_vector: Dense query embedding (same dimension as stored vecs).
            top_k:        Maximum results to return.
            filters:      Optional metadata equality filters (backend-specific).

        Returns:
            List of ``SearchResult`` objects sorted by descending similarity.
        """

    @abstractmethod
    def delete_by_file(self, source_file_id: str) -> int:
        """Delete all chunks belonging to *source_file_id*.

        Used when a file is deleted or re-indexed to avoid stale chunks.

        Args:
            source_file_id: The ``SourceFile.id`` to purge.

        Returns:
            Number of chunks deleted.
        """

    @abstractmethod
    def get_stats(self) -> IndexStats:
        """Return summary statistics for the current collection.

        Returns:
            An ``IndexStats`` object.
        """

    @abstractmethod
    def clear(self) -> None:
        """Delete all data in the collection.  Use with caution."""

    @property
    def name(self) -> str:
        """Human-readable name of this vector store backend."""
        return f"{type(self).__name__}({self.collection_name})"
