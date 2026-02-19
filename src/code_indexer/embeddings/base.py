"""
Abstract base class for embedding backends.

An embedder takes a list of text strings and returns a corresponding list of
dense floating-point vectors.  Callers should always use the batch API even
for single strings to allow backends to optimise (e.g. GPU batching).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from code_indexer.core.models import Chunk, EmbeddedChunk


class BaseEmbedder(ABC):
    """Abstract embedder interface.

    Args:
        model:      Model identifier (backend-specific).
        batch_size: Maximum number of texts per API/model call.
        dimensions: Expected vector dimensionality.
    """

    def __init__(
        self,
        model: str,
        batch_size: int = 64,
        dimensions: int = 384,
    ) -> None:
        self.model = model
        self.batch_size = batch_size
        self.dimensions = dimensions

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of text strings.

        Args:
            texts: List of strings to embed.  May be empty.

        Returns:
            List of float vectors, same length as *texts*.  Each vector
            has ``self.dimensions`` elements.

        Raises:
            EmbeddingError: On backend failures.
        """

    def embed_chunks(self, chunks: list[Chunk]) -> list[EmbeddedChunk]:
        """Embed a list of ``Chunk`` objects.

        Converts each chunk to its embedding text representation, calls
        ``embed_texts`` in batches, and returns ``EmbeddedChunk`` objects.

        Args:
            chunks: List of chunks to embed.

        Returns:
            List of ``EmbeddedChunk`` objects in the same order.
        """
        texts = [chunk.to_embedding_text() for chunk in chunks]
        embeddings = self.embed_texts(texts)
        return [
            EmbeddedChunk(chunk=chunk, embedding=emb, model=self.model)
            for chunk, emb in zip(chunks, embeddings)
        ]

    @property
    def name(self) -> str:
        """Human-readable name of this embedder."""
        return f"{type(self).__name__}({self.model})"
