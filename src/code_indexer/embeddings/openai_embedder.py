"""
OpenAI embedding backend.

Uses the ``openai`` Python SDK to call the Embeddings API.

Models (as of 2024)
--------------------
* ``text-embedding-3-small`` – 1536 dimensions, cost-efficient, good quality.
* ``text-embedding-3-large`` – 3072 dimensions, highest quality.
* ``text-embedding-ada-002``  – 1536 dimensions, legacy but still supported.

Rate limiting
--------------
The OpenAI Embeddings API has per-minute token limits (not request limits).
We use ``tenacity`` for exponential-backoff retries on 429 and 5xx errors.
Large batches are automatically split into ``batch_size`` sub-batches.
"""

from __future__ import annotations

import logging

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from code_indexer.core.exceptions import EmbeddingProviderError, EmbeddingRateLimitError
from code_indexer.embeddings.base import BaseEmbedder

logger = logging.getLogger(__name__)


class OpenAIEmbedder(BaseEmbedder):
    """Embed text using the OpenAI Embeddings API.

    Args:
        model:       OpenAI model name.
        api_key:     OpenAI API key.  Falls back to the ``OPENAI_API_KEY``
                     environment variable when empty.
        batch_size:  Maximum texts per API call.  OpenAI allows up to 2048
                     inputs per call; we default to 256 for reliability.
        dimensions:  Expected output dimension.  For ``text-embedding-3-*``
                     models you can pass ``dimensions`` to truncate.
        max_retries: Maximum number of retry attempts on transient errors.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str = "",
        batch_size: int = 256,
        dimensions: int = 1536,
        max_retries: int = 5,
    ) -> None:
        super().__init__(model=model, batch_size=batch_size, dimensions=dimensions)
        self._api_key = api_key
        self._max_retries = max_retries
        self._client: object | None = None  # lazy init

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed *texts* via the OpenAI Embeddings API.

        Args:
            texts: List of strings to embed.

        Returns:
            List of float vectors (same order as input).

        Raises:
            EmbeddingRateLimitError: On HTTP 429.
            EmbeddingProviderError:  On other API errors.
        """
        if not texts:
            return []

        client = self._get_client()
        all_embeddings: list[list[float]] = []

        # Split into batches.
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            embeddings = self._call_api(client, batch)
            all_embeddings.extend(embeddings)

        return all_embeddings

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_client(self) -> object:
        """Return (or lazily initialise) the OpenAI client."""
        if self._client is None:
            try:
                import openai  # noqa: PLC0415

                kwargs: dict = {}
                if self._api_key:
                    kwargs["api_key"] = self._api_key
                self._client = openai.OpenAI(**kwargs)
            except ImportError as exc:
                raise EmbeddingProviderError(
                    "openai package not installed. Run: pip install openai"
                ) from exc
        return self._client

    @retry(
        retry=retry_if_exception_type(Exception),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def _call_api(self, client: object, texts: list[str]) -> list[list[float]]:
        """Make a single batched API call with retry logic.

        Args:
            client: OpenAI client.
            texts:  Texts to embed (one batch).

        Returns:
            List of float vectors.
        """
        try:
            kwargs: dict = {"input": texts, "model": self.model}
            # For text-embedding-3-* models, we can request specific dims.
            if "text-embedding-3" in self.model:
                kwargs["dimensions"] = self.dimensions

            response = client.embeddings.create(**kwargs)  # type: ignore[union-attr]
            return [item.embedding for item in response.data]

        except Exception as exc:
            exc_str = str(exc).lower()
            if "rate limit" in exc_str or "429" in exc_str:
                raise EmbeddingRateLimitError(
                    f"OpenAI rate limit hit: {exc}"
                ) from exc
            raise EmbeddingProviderError(f"OpenAI API error: {exc}") from exc
