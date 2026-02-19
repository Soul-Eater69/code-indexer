"""
In-memory vector store backed by NumPy brute-force search.

This store holds all vectors in RAM and performs an exhaustive cosine
similarity scan on every query.  It is suitable for:
  * Unit tests (no external dependencies).
  * Small codebases (<50k chunks, ~200MB RAM for 384-dim vectors).
  * CLI one-shot index+query workflows where persistence is not needed.

For production use with large codebases, prefer ChromaDB or Qdrant.

Cosine similarity formula:
  sim(a, b) = (a · b) / (||a|| × ||b||)

Since we normalise all stored vectors at insertion time, the dot product
alone gives cosine similarity and the division is not needed at query time.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from datetime import datetime, timezone

from code_indexer.core.models import (
    Chunk,
    EmbeddedChunk,
    IndexStats,
    SearchResult,
)
from code_indexer.vectorstore.base import BaseVectorStore

logger = logging.getLogger(__name__)


def _l2_norm(vec: list[float]) -> list[float]:
    """Return the L2-normalised version of *vec*."""
    magnitude = math.sqrt(sum(x * x for x in vec))
    if magnitude == 0:
        return vec
    return [x / magnitude for x in vec]


def _dot(a: list[float], b: list[float]) -> float:
    """Return the dot product of two equal-length vectors."""
    return sum(x * y for x, y in zip(a, b))


class InMemoryVectorStore(BaseVectorStore):
    """Thread-unsafe in-memory vector store (suitable for single-thread use).

    Args:
        collection_name: Logical name (not used for storage but kept for
                         API compatibility).
        dimensions:      Expected vector dimensionality.
        distance_metric: Only ``"cosine"`` is implemented; other values are
                         silently treated as cosine.
    """

    def __init__(
        self,
        collection_name: str = "code_index",
        dimensions: int = 384,
        distance_metric: str = "cosine",
    ) -> None:
        super().__init__(
            collection_name=collection_name,
            dimensions=dimensions,
            distance_metric=distance_metric,
        )
        # chunk_id → (normalised_vector, EmbeddedChunk)
        self._store: dict[str, tuple[list[float], EmbeddedChunk]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def upsert(self, embedded_chunks: list[EmbeddedChunk]) -> int:
        """Store *embedded_chunks* in memory.

        Vectors are L2-normalised at insertion time so that dot-product at
        query time equals cosine similarity.

        Args:
            embedded_chunks: Chunks with vectors.

        Returns:
            Number of chunks stored.
        """
        for ec in embedded_chunks:
            normalised = _l2_norm(ec.embedding)
            self._store[ec.chunk.id] = (normalised, ec)

        logger.debug("InMemoryStore: %d chunks stored (total=%d)", len(embedded_chunks), len(self._store))
        return len(embedded_chunks)

    def query(
        self,
        query_vector: list[float],
        top_k: int = 10,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        """Return the top-k most similar chunks via brute-force cosine search.

        Args:
            query_vector: Dense query vector (normalised internally).
            top_k:        Max results.
            filters:      Dict of metadata field → value for exact-match
                          filtering.  Supported fields: ``language``,
                          ``chunk_type``, ``path`` (prefix match), ``name``.

        Returns:
            Sorted list of ``SearchResult`` (best first).
        """
        if not self._store:
            return []

        norm_query = _l2_norm(query_vector)
        scored: list[tuple[float, EmbeddedChunk]] = []

        for norm_vec, ec in self._store.values():
            if filters and not self._matches_filters(ec.chunk, filters):
                continue
            score = _dot(norm_query, norm_vec)
            scored.append((score, ec))

        # Sort descending by score.
        scored.sort(key=lambda t: t[0], reverse=True)
        top = scored[:top_k]

        return [
            SearchResult(chunk=ec.chunk, score=score, rank=rank + 1)
            for rank, (score, ec) in enumerate(top)
        ]

    def delete_by_file(self, source_file_id: str) -> int:
        """Delete all chunks belonging to *source_file_id*.

        Args:
            source_file_id: Source file ID to purge.

        Returns:
            Number of chunks removed.
        """
        to_delete = [
            chunk_id
            for chunk_id, (_, ec) in self._store.items()
            if ec.chunk.source_file_id == source_file_id
        ]
        for chunk_id in to_delete:
            del self._store[chunk_id]
        return len(to_delete)

    def get_stats(self) -> IndexStats:
        """Return summary statistics for the in-memory store."""
        languages: dict[str, int] = defaultdict(int)
        chunk_types: dict[str, int] = defaultdict(int)
        total_tokens = 0
        file_ids: set[str] = set()

        for _, ec in self._store.values():
            chunk = ec.chunk
            languages[chunk.language.value] += 1
            chunk_types[chunk.chunk_type.value] += 1
            total_tokens += chunk.token_count
            file_ids.add(chunk.source_file_id)

        return IndexStats(
            total_files=len(file_ids),
            total_chunks=len(self._store),
            total_tokens=total_tokens,
            languages=dict(languages),
            chunk_types=dict(chunk_types),
            vector_store="in_memory",
        )

    def clear(self) -> None:
        """Remove all data from the store."""
        self._store.clear()
        logger.info("InMemoryStore cleared")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _matches_filters(chunk: Chunk, filters: dict) -> bool:
        """Return ``True`` if *chunk* satisfies all *filters*.

        Args:
            chunk:   The chunk to test.
            filters: Dict of field → expected value.
        """
        for key, value in filters.items():
            if key == "language" and chunk.language.value != value:
                return False
            if key == "chunk_type" and chunk.chunk_type.value != value:
                return False
            if key == "path_prefix" and not chunk.path.startswith(str(value)):
                return False
            if key == "name" and chunk.name != value:
                return False
        return True
