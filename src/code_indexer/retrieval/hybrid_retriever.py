"""Hybrid retriever combining vector similarity search with graph traversal.

Two-stage retrieval
-------------------
Stage 1 — **Vector search** (semantic similarity, :class:`CodeRetriever`):
  Embeds the query and performs ANN search.  Returns up to ``top_k * 2``
  candidates to give the graph expansion stage something to work with.

Stage 2 — **Graph expansion** (:class:`BaseGraphStore`):
  For each vector-search result, looks up the corresponding
  :class:`~code_indexer.graph.models.SymbolContext` and adds:

  * Direct **callers** and **callees** (one hop).
  * **Parent class** (if the result is a method).
  * **Sibling methods** of the parent class.
  * **Base classes** (one level of inheritance).

  The expanded chunks are de-duplicated and re-ranked by a combined score::

      combined_score = vector_score * vector_weight
                     + graph_score  * (1 - vector_weight)

  where ``graph_score`` is a fixed value assigned to graph-expanded results
  (default ``0.6``) to indicate that they are structurally related but
  not directly similar by embedding distance.

Usage::

    retriever = HybridRetriever(
        embedder=my_embedder,
        vector_store=my_vector_store,
        graph_store=my_graph_store,
        vector_weight=0.7,
    )
    response = retriever.search(SearchQuery(
        query="function that verifies JWT tokens",
        top_k=5,
    ))
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from code_indexer.core.models import SearchQuery, SearchResponse, SearchResult
from code_indexer.retrieval.retriever import CodeRetriever

if TYPE_CHECKING:
    from code_indexer.embeddings.base import BaseEmbedder
    from code_indexer.graph.base import BaseGraphStore
    from code_indexer.vectorstore.base import BaseVectorStore

logger = logging.getLogger(__name__)

# Score assigned to graph-expanded chunks that did not appear in vector results.
_GRAPH_EXPANSION_SCORE = 0.6


class HybridRetriever:
    """Vector-similarity + graph-structure combined retrieval.

    Parameters
    ----------
    embedder:
        Embedding backend (same model used to build the index).
    vector_store:
        Vector store holding the embedded chunks.
    graph_store:
        Code knowledge graph store.
    vector_weight:
        Weight for vector similarity score in [0, 1].
        ``1.0`` = pure vector search.  ``0.0`` = pure graph expansion.
    graph_expansion_hops:
        How many hops to traverse in the call graph for expansion.
        ``1`` = direct callers/callees only.
    max_graph_expansions:
        Maximum number of additional chunks to add via graph expansion
        per vector result.
    """

    def __init__(
        self,
        embedder: "BaseEmbedder",
        vector_store: "BaseVectorStore",
        graph_store: "BaseGraphStore",
        *,
        vector_weight: float = 0.7,
        graph_expansion_hops: int = 1,
        max_graph_expansions: int = 3,
    ) -> None:
        self._vector_retriever = CodeRetriever(
            embedder=embedder,
            vector_store=vector_store,
        )
        self._graph = graph_store
        self._vector_weight = vector_weight
        self._graph_weight = 1.0 - vector_weight
        self._expansion_hops = graph_expansion_hops
        self._max_expansions = max_graph_expansions

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(self, query: SearchQuery) -> SearchResponse:
        """Perform hybrid search.

        Steps:

        1. Run vector search for ``top_k * 2`` candidates.
        2. For each result, expand via graph (callers, callees, parent class).
        3. De-duplicate and re-score.
        4. Return the top ``top_k`` results.
        """
        t0 = time.perf_counter()

        # --- Stage 1: vector search ---
        vector_query = query.model_copy(update={"top_k": query.top_k * 2})
        vector_response = self._vector_retriever.search(vector_query)

        # --- Stage 2: graph expansion ---
        expanded = self._expand_with_graph(vector_response.results, query)

        # --- Stage 3: de-duplicate and re-rank ---
        final = self._merge_and_rank(vector_response.results, expanded, query.top_k)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        return SearchResponse(
            query=query.query,
            results=final,
            total=len(final),
            latency_ms=round(elapsed_ms, 2),
        )

    def get_symbol_context_text(self, symbol_name: str) -> str | None:
        """Return a structural context string for a symbol by name.

        Useful for injecting into LLM prompts alongside the code chunk.
        Returns ``None`` if the symbol is not in the graph.
        """
        symbols = self._graph.find_symbols_by_name(symbol_name)
        if not symbols:
            return None
        ctx = self._graph.get_symbol_context(symbols[0].id)
        if ctx is None:
            return None
        return ctx.to_context_text()

    # ------------------------------------------------------------------
    # Graph expansion helpers
    # ------------------------------------------------------------------

    def _expand_with_graph(
        self,
        vector_results: list[SearchResult],
        query: SearchQuery,
    ) -> list[SearchResult]:
        """Expand each vector result by fetching related symbols from the graph."""
        extra: list[SearchResult] = []
        seen_chunk_ids: set[str] = {r.chunk.id for r in vector_results}

        for vr in vector_results[: query.top_k]:  # only expand top-k originals
            chunk = vr.chunk
            # Try to find the symbol in the graph by matching name + file
            symbols = self._graph.find_symbols_by_name(
                chunk.name or "", file_path=chunk.path
            )
            if not symbols:
                continue

            sym = symbols[0]
            ctx = self._graph.get_symbol_context(sym.id)
            if ctx is None:
                continue

            # Collect related symbols (callers, callees, parent, siblings)
            related_symbols = []
            related_symbols.extend(ctx.callers[: self._max_expansions])
            related_symbols.extend(ctx.callees[: self._max_expansions])
            if ctx.parent_class:
                related_symbols.append(ctx.parent_class)
            related_symbols.extend(ctx.methods[: self._max_expansions])
            related_symbols.extend(ctx.inherits_from)

            # For each related symbol, try to fetch its chunk from the vector store
            expansions_added = 0
            for rel_sym in related_symbols:
                if expansions_added >= self._max_expansions:
                    break
                # Fetch chunks that match this symbol's file + name
                related_chunks = self._vector_retriever._vector_store.query(
                    query_vector=self._vector_retriever._last_query_vector or [],
                    top_k=3,
                    filters={"path": rel_sym.file_path, "name": rel_sym.name},
                ) if hasattr(self._vector_retriever, "_last_query_vector") else []

                for related_result in related_chunks:
                    if related_result.chunk.id in seen_chunk_ids:
                        continue
                    seen_chunk_ids.add(related_result.chunk.id)
                    # Assign a combined score: lower than direct match
                    graph_result = SearchResult(
                        chunk=related_result.chunk,
                        score=_GRAPH_EXPANSION_SCORE * self._graph_weight
                        + related_result.score * self._vector_weight,
                        rank=0,  # will be reassigned in merge
                    )
                    extra.append(graph_result)
                    expansions_added += 1

        return extra

    def _merge_and_rank(
        self,
        vector_results: list[SearchResult],
        graph_results: list[SearchResult],
        top_k: int,
    ) -> list[SearchResult]:
        """Merge vector + graph results, re-weight scores, and return top_k."""
        # Adjust vector result scores
        weighted: list[SearchResult] = []
        seen: set[str] = set()

        for r in vector_results:
            if r.chunk.id in seen:
                continue
            seen.add(r.chunk.id)
            weighted.append(
                SearchResult(
                    chunk=r.chunk,
                    score=r.score * self._vector_weight,
                    rank=0,
                )
            )

        for r in graph_results:
            if r.chunk.id in seen:
                continue
            seen.add(r.chunk.id)
            weighted.append(r)

        # Sort by combined score descending
        weighted.sort(key=lambda r: r.score, reverse=True)
        top = weighted[:top_k]

        # Assign final ranks
        for i, r in enumerate(top):
            r.rank = i + 1

        return top
