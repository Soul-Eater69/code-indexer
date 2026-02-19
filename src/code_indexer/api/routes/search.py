"""
Search API routes.

Endpoints:
  POST /search           – Semantic similarity search.
  POST /search/rerank    – Search with cross-encoder reranking.
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException, Request

from code_indexer.core.models import SearchQuery, SearchResponse
from code_indexer.retrieval.retriever import CodeRetriever

router = APIRouter()


@router.post("", response_model=SearchResponse)
async def search(query: SearchQuery, request: Request) -> SearchResponse:
    """Perform a semantic similarity search over the indexed codebase.

    The query is embedded using the same model as the index, then the
    top-k most similar chunks are retrieved from the vector store.

    Example request::

        POST /search
        {
          "query": "function that validates JWT tokens",
          "top_k": 5,
          "language": "python",
          "chunk_types": ["function"],
          "min_score": 0.3,
          "include_context": true
        }
    """
    embedder = request.app.state.embedder
    vector_store = request.app.state.vector_store

    if embedder is None:
        raise HTTPException(status_code=503, detail="Embedder not configured")
    if vector_store is None:
        raise HTTPException(status_code=503, detail="Vector store not configured")

    retriever = CodeRetriever(embedder=embedder, vector_store=vector_store)

    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(None, lambda: retriever.search(query))
    return response


@router.post("/rerank", response_model=SearchResponse)
async def search_with_rerank(query: SearchQuery, request: Request) -> SearchResponse:
    """Search and rerank results using a cross-encoder model.

    Retrieves ``top_k * 2`` candidates from the vector store, then applies
    a cross-encoder reranker to produce the final ``top_k`` results.

    Cross-encoder reranking produces better precision at the cost of
    additional inference time (~100-200ms for top-20 candidates on CPU).
    """
    embedder = request.app.state.embedder
    vector_store = request.app.state.vector_store

    if embedder is None:
        raise HTTPException(status_code=503, detail="Embedder not configured")
    if vector_store is None:
        raise HTTPException(status_code=503, detail="Vector store not configured")

    from code_indexer.retrieval.reranker import CrossEncoderReranker  # noqa: PLC0415

    reranker = CrossEncoderReranker()
    retriever = CodeRetriever(
        embedder=embedder,
        vector_store=vector_store,
        reranker=reranker,
    )

    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(None, lambda: retriever.search(query))
    return response
