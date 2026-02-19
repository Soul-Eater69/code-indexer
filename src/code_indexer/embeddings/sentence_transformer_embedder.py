"""
Sentence-Transformers embedding backend.

This is the recommended default backend for self-hosted deployments.  It runs
entirely locally (no API key needed, no data leaves your machine) and supports
GPU acceleration.

Model recommendations for code
--------------------------------
* ``all-MiniLM-L6-v2`` – 384 dims, very fast, good general-purpose baseline.
* ``all-mpnet-base-v2`` – 768 dims, better quality but 3× slower.
* ``microsoft/codebert-base`` – 768 dims, fine-tuned on code pairs, excellent
  for code similarity tasks.
* ``Salesforce/codet5p-110m-embedding`` – 256 dims, tiny & fast code model.

Batch inference note
---------------------
``sentence_transformers`` handles internal batching automatically when you
call ``model.encode(list_of_texts)``.  We still honour our configured
``batch_size`` to cap peak memory usage.
"""

from __future__ import annotations

import logging

from code_indexer.core.exceptions import EmbeddingProviderError
from code_indexer.embeddings.base import BaseEmbedder

logger = logging.getLogger(__name__)


class SentenceTransformerEmbedder(BaseEmbedder):
    """Embed text using a local Sentence-Transformers model.

    The model is downloaded from HuggingFace Hub on first use and cached
    in ``~/.cache/huggingface``.

    Args:
        model:      HuggingFace model identifier.
        batch_size: Number of texts per encode call.
        device:     PyTorch device string (``"cpu"``, ``"cuda"``,
                    ``"mps"``).  ``None`` means auto-detect.
        normalize:  Whether to L2-normalise output embeddings.
                    Required for cosine-similarity distance metrics.
    """

    def __init__(
        self,
        model: str = "all-MiniLM-L6-v2",
        batch_size: int = 64,
        device: str | None = None,
        normalize: bool = True,
    ) -> None:
        # Dimension is resolved lazily after the model is loaded.
        super().__init__(model=model, batch_size=batch_size, dimensions=0)
        self._device = device
        self._normalize = normalize
        self._model: object | None = None

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed *texts* using the local Sentence-Transformers model.

        Args:
            texts: Input strings.

        Returns:
            List of float vectors.

        Raises:
            EmbeddingProviderError: If the package is not installed or the
                                    model cannot be loaded.
        """
        if not texts:
            return []

        model = self._get_model()
        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            try:
                vecs = model.encode(  # type: ignore[union-attr]
                    batch,
                    normalize_embeddings=self._normalize,
                    show_progress_bar=False,
                )
                all_embeddings.extend(vecs.tolist())
            except Exception as exc:
                raise EmbeddingProviderError(
                    f"SentenceTransformer encode failed: {exc}"
                ) from exc

        return all_embeddings

    def _get_model(self) -> object:
        """Lazily load and cache the Sentence-Transformers model."""
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer  # noqa: PLC0415

            logger.info("Loading Sentence-Transformers model %r …", self.model)
            kwargs: dict = {}
            if self._device:
                kwargs["device"] = self._device
            self._model = SentenceTransformer(self.model, **kwargs)
            # Update dimensions from the loaded model.
            self.dimensions = self._model.get_sentence_embedding_dimension()  # type: ignore[union-attr]
            logger.info(
                "Model loaded: %s (%d dims)", self.model, self.dimensions
            )
            return self._model
        except ImportError as exc:
            raise EmbeddingProviderError(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers"
            ) from exc
