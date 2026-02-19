"""
Ollama embedding backend.

Ollama (https://ollama.ai) is a local LLM server that can serve embedding
models.  This backend sends HTTP requests to a running Ollama instance.

Useful models
--------------
* ``nomic-embed-text``  – 768 dims, fast, recommended for text.
* ``mxbai-embed-large`` – 1024 dims, high quality.
* ``codellama``         – Can be used for code embeddings via the /api/embeddings
                          endpoint (experimental).

Pull a model first::

    ollama pull nomic-embed-text
"""

from __future__ import annotations

import logging

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from code_indexer.core.exceptions import EmbeddingProviderError
from code_indexer.embeddings.base import BaseEmbedder

logger = logging.getLogger(__name__)


class OllamaEmbedder(BaseEmbedder):
    """Embed text using a locally running Ollama server.

    Args:
        model:       Ollama model tag (e.g. ``"nomic-embed-text"``).
        base_url:    Base URL of the Ollama server.
        batch_size:  Number of texts per call (Ollama processes one at a
                     time so batching is sequential on our side).
        dimensions:  Expected vector dimensionality.
        timeout:     HTTP request timeout in seconds.
    """

    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
        batch_size: int = 32,
        dimensions: int = 768,
        timeout: float = 30.0,
    ) -> None:
        super().__init__(model=model, batch_size=batch_size, dimensions=dimensions)
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed *texts* by calling the Ollama ``/api/embeddings`` endpoint.

        Args:
            texts: Input strings.

        Returns:
            List of float vectors.

        Raises:
            EmbeddingProviderError: On connection errors or non-200 responses.
        """
        if not texts:
            return []

        all_embeddings: list[list[float]] = []
        with httpx.Client(timeout=self._timeout) as client:
            for text in texts:
                embedding = self._call_ollama(client, text)
                all_embeddings.append(embedding)

        return all_embeddings

    @retry(
        retry=retry_if_exception_type(httpx.TransportError),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def _call_ollama(self, client: httpx.Client, text: str) -> list[float]:
        """Call ``/api/embeddings`` for a single text string.

        Args:
            client: HTTP client.
            text:   Text to embed.

        Returns:
            Float vector.
        """
        try:
            response = client.post(
                f"{self._base_url}/api/embeddings",
                json={"model": self.model, "prompt": text},
            )
            response.raise_for_status()
            data = response.json()
            return data["embedding"]
        except httpx.HTTPStatusError as exc:
            raise EmbeddingProviderError(
                f"Ollama returned HTTP {exc.response.status_code}: {exc.response.text}"
            ) from exc
        except (KeyError, ValueError) as exc:
            raise EmbeddingProviderError(
                f"Unexpected Ollama response format: {exc}"
            ) from exc
