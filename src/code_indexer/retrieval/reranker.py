"""
Reranking: improve result quality after vector search.

Why rerank?
-----------
Bi-encoder retrieval (embedding query + ANN search) is fast but approximate.
A cross-encoder model reads the query *and* the document together and produces
a much better relevance score, at the cost of O(k) forward passes.

We use a two-stage pipeline (bi-encoder → cross-encoder):
  1. Retrieve top-k*2 candidates via bi-encoder (fast, recall-oriented).
  2. Rerank with a cross-encoder on the smaller candidate set (accurate, precision-oriented).

Cross-encoder models for code
-------------------------------
* ``cross-encoder/ms-marco-MiniLM-L-6-v2`` – General text, fast.
* ``jinaai/jina-reranker-v2-base-multilingual`` – Code-aware, higher quality.
* ``BAAI/bge-reranker-base`` – Good for code, widely used.

Fallback: ``ScoreReranker`` simply re-orders by the original bi-encoder score
(i.e. a no-op) and is used when no cross-encoder is configured.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from code_indexer.core.models import SearchResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------


class BaseReranker(ABC):
    """Abstract reranker interface."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int,
    ) -> list[SearchResult]:
        """Rerank *results* with respect to *query*.

        Args:
            query:   Original query string.
            results: Candidate results from vector search.
            top_k:   How many results to return after reranking.

        Returns:
            Reranked and truncated list (best first).
        """


# ---------------------------------------------------------------------------
# Score-based no-op reranker (default)
# ---------------------------------------------------------------------------


class ScoreReranker(BaseReranker):
    """Trivial reranker that just sorts by the original bi-encoder score.

    This is the identity transformation: no cross-encoder inference is run.
    Use it as a baseline or when no GPU/cross-encoder is available.
    """

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int,
    ) -> list[SearchResult]:
        return sorted(results, key=lambda r: r.score, reverse=True)[:top_k]


# ---------------------------------------------------------------------------
# Cross-encoder reranker
# ---------------------------------------------------------------------------


class CrossEncoderReranker(BaseReranker):
    """Rerank using a ``sentence_transformers.CrossEncoder`` model.

    Args:
        model:   HuggingFace cross-encoder model name.
        device:  PyTorch device (``"cpu"``, ``"cuda"``, ``"mps"``).
        max_length: Maximum sequence length for the cross-encoder.
    """

    def __init__(
        self,
        model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str | None = None,
        max_length: int = 512,
    ) -> None:
        self._model_name = model
        self._device = device
        self._max_length = max_length
        self._model: object | None = None

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int,
    ) -> list[SearchResult]:
        """Score all (query, document) pairs and return top-k.

        Args:
            query:   Query string.
            results: Candidate results.
            top_k:   Number of results to return.

        Returns:
            Reranked list with updated scores.
        """
        if not results:
            return []

        model = self._get_model()
        pairs = [(query, r.chunk.content) for r in results]

        try:
            scores = model.predict(pairs, show_progress_bar=False)  # type: ignore[union-attr]
        except Exception as exc:
            logger.error("CrossEncoder predict failed: %s", exc)
            return results[:top_k]

        # Pair scores with results and sort.
        scored = sorted(
            zip(scores, results),
            key=lambda t: float(t[0]),
            reverse=True,
        )

        reranked: list[SearchResult] = []
        for rank, (score, result) in enumerate(scored[:top_k], start=1):
            result.score = float(score)
            result.rank = rank
            reranked.append(result)

        logger.debug("CrossEncoder reranked %d → %d results", len(results), len(reranked))
        return reranked

    def _get_model(self) -> object:
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import CrossEncoder  # noqa: PLC0415

            logger.info("Loading CrossEncoder %r …", self._model_name)
            kwargs: dict = {"max_length": self._max_length}
            if self._device:
                kwargs["device"] = self._device
            self._model = CrossEncoder(self._model_name, **kwargs)
            return self._model
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers"
            ) from exc
