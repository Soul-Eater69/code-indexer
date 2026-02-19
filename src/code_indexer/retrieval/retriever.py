"""
Retrieval layer: query the index and return ranked results.

The retriever ties together:
  1. Embedding the query string using the same model used during indexing.
  2. Calling the vector store for approximate nearest-neighbour search.
  3. Applying metadata filters from the ``SearchQuery``.
  4. Optionally reranking results with a cross-encoder.
  5. Wrapping results in a ``SearchResponse``.

Retrieval is the read-path of the system.  It is intentionally separate from
the indexing (write) path so the two can be scaled independently.

Usage::

    retriever = CodeRetriever(embedder=embedder, vector_store=store)
    response = retriever.search(SearchQuery(query="parse JWT token", top_k=5))
    for result in response.results:
        print(result.chunk.path, result.score)
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from code_indexer.core.models import (
    ChunkType,
    Language,
    SearchQuery,
    SearchResponse,
    SearchResult,
)
from code_indexer.embeddings.base import BaseEmbedder
from code_indexer.vectorstore.base import BaseVectorStore

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class CodeRetriever:
    """Query the code index and return ranked results.

    Args:
        embedder:      Embedding backend (must match the one used during
                       indexing).
        vector_store:  The populated vector store.
        reranker:      Optional reranker to apply after vector search.
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        vector_store: BaseVectorStore,
        reranker: "BaseReranker | None" = None,
    ) -> None:
        self._embedder = embedder
        self._vector_store = vector_store
        self._reranker = reranker

    def search(self, query: SearchQuery) -> SearchResponse:
        """Execute a vector similarity search.

        Args:
            query: Search parameters including the query string and filters.

        Returns:
            ``SearchResponse`` with ordered results.
        """
        start = time.perf_counter()

        # 1. Embed the query string.
        query_vectors = self._embedder.embed_texts([query.query])
        query_vector = query_vectors[0]

        # 2. Build vector store filters from the SearchQuery.
        vs_filters = self._build_filters(query)

        # 3. Retrieve from vector store.
        # We over-fetch by 2× when reranking to give the reranker more candidates.
        fetch_k = query.top_k * 2 if self._reranker else query.top_k
        raw_results = self._vector_store.query(
            query_vector=query_vector,
            top_k=fetch_k,
            filters=vs_filters or None,
        )

        # 4. Post-filter by min_score.
        if query.min_score > 0:
            raw_results = [r for r in raw_results if r.score >= query.min_score]

        # 5. Optionally strip context.
        if not query.include_context:
            for result in raw_results:
                result.chunk.context_before = ""
                result.chunk.context_after = ""

        # 6. Rerank if configured.
        if self._reranker and raw_results:
            raw_results = self._reranker.rerank(query.query, raw_results, query.top_k)
        else:
            raw_results = raw_results[: query.top_k]

        # 7. Assign final ranks.
        for rank, result in enumerate(raw_results, start=1):
            result.rank = rank

        elapsed_ms = (time.perf_counter() - start) * 1000

        logger.debug(
            "Search %r: %d results in %.1f ms",
            query.query,
            len(raw_results),
            elapsed_ms,
        )

        return SearchResponse(
            query=query.query,
            results=raw_results,
            total=len(raw_results),
            latency_ms=elapsed_ms,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_filters(query: SearchQuery) -> dict:
        """Translate ``SearchQuery`` filter fields to a flat dict for the store.

        The in-memory and ChromaDB stores accept flat ``{field: value}`` dicts.
        Qdrant's store translates these internally to its own ``Filter`` model.

        Args:
            query: The search query.

        Returns:
            Dict of metadata filters (may be empty).
        """
        filters: dict = {}

        if query.language:
            filters["language"] = query.language.value

        if query.chunk_types and len(query.chunk_types) == 1:
            # Single chunk type: simple equality filter.
            filters["chunk_type"] = query.chunk_types[0].value
        # Multiple chunk types require OR logic; left for vector store to handle.

        if query.path_prefix:
            filters["path_prefix"] = query.path_prefix

        filters.update(query.metadata_filters)

        return filters
