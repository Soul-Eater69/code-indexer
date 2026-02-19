"""
ChromaDB vector store backend.

ChromaDB is an open-source embedding database that runs in-process (no server
needed) with optional on-disk persistence.  It is the recommended default for
local / development setups.

Storage modes:
  * In-memory: ``persist_directory=None`` – data is lost on process exit.
  * On-disk:   ``persist_directory=".chroma"`` – survives restarts.
  * Client-server: ``host/port`` – connect to a standalone Chroma server
    (not implemented here; use ``qdrant_store.py`` for production).

Data model inside ChromaDB:
  - Each ``EmbeddedChunk`` becomes one ChromaDB document.
  - The document ``id`` = ``chunk.id`` (deterministic SHA-256 prefix).
  - The ``embedding`` is the float vector.
  - Metadata is stored as a flat dict of primitive values.
  - The document text is ``chunk.to_embedding_text()``.

Filtering:
  ChromaDB uses ``where`` dicts for metadata filtering::

      {"language": "python"}
      {"$and": [{"language": "python"}, {"chunk_type": "function"}]}
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from code_indexer.core.exceptions import VectorStoreError
from code_indexer.core.models import (
    Chunk,
    ChunkType,
    EmbeddedChunk,
    IndexStats,
    Language,
    SearchResult,
)
from code_indexer.vectorstore.base import BaseVectorStore

logger = logging.getLogger(__name__)


class ChromaVectorStore(BaseVectorStore):
    """Store and retrieve code embeddings using ChromaDB.

    Args:
        collection_name:  Name of the ChromaDB collection.
        persist_directory: Path for on-disk persistence.  ``None`` for
                           in-memory (ephemeral) mode.
        dimensions:       Embedding dimensionality.
        distance_metric:  ``"cosine"``, ``"l2"``, or ``"ip"`` (inner product).
    """

    _METRIC_MAP = {"cosine": "cosine", "euclidean": "l2", "dot": "ip"}

    def __init__(
        self,
        collection_name: str = "code_index",
        persist_directory: str | None = ".chroma",
        dimensions: int = 384,
        distance_metric: str = "cosine",
    ) -> None:
        super().__init__(
            collection_name=collection_name,
            dimensions=dimensions,
            distance_metric=distance_metric,
        )
        self._persist_directory = persist_directory
        self._client: Any = None
        self._collection: Any = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def upsert(self, embedded_chunks: list[EmbeddedChunk]) -> int:
        """Insert or update *embedded_chunks* in ChromaDB.

        Args:
            embedded_chunks: Chunks with vectors.

        Returns:
            Number of chunks stored.
        """
        if not embedded_chunks:
            return 0

        col = self._get_collection()
        ids, embeddings, metadatas, documents = [], [], [], []

        for ec in embedded_chunks:
            chunk = ec.chunk
            ids.append(chunk.id)
            embeddings.append(ec.embedding)
            documents.append(chunk.to_embedding_text())
            metadatas.append(self._chunk_to_meta(chunk, ec.model))

        try:
            col.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )
            logger.debug("ChromaDB upserted %d chunks", len(ids))
            return len(ids)
        except Exception as exc:
            raise VectorStoreError(f"ChromaDB upsert failed: {exc}") from exc

    def query(
        self,
        query_vector: list[float],
        top_k: int = 10,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        """Query ChromaDB for the nearest neighbours of *query_vector*.

        Args:
            query_vector: Dense query embedding.
            top_k:        Max results.
            filters:      ChromaDB ``where`` dict (metadata filters).

        Returns:
            List of ``SearchResult`` sorted by descending similarity.
        """
        col = self._get_collection()
        kwargs: dict = {
            "query_embeddings": [query_vector],
            "n_results": top_k,
            "include": ["metadatas", "distances", "documents"],
        }
        if filters:
            kwargs["where"] = filters

        try:
            results = col.query(**kwargs)
        except Exception as exc:
            raise VectorStoreError(f"ChromaDB query failed: {exc}") from exc

        search_results: list[SearchResult] = []
        ids = results["ids"][0]
        distances = results["distances"][0]
        metadatas = results["metadatas"][0]
        documents = results["documents"][0]

        for idx, (chunk_id, dist, meta, doc) in enumerate(
            zip(ids, distances, metadatas, documents)
        ):
            chunk = self._meta_to_chunk(chunk_id, meta, doc)
            # ChromaDB returns L2 distance for cosine metric (it normalises
            # internally), so distance ∈ [0, 2].  Convert to similarity ∈ [0, 1].
            score = max(0.0, 1.0 - dist / 2.0)
            search_results.append(SearchResult(chunk=chunk, score=score, rank=idx + 1))

        return search_results

    def delete_by_file(self, source_file_id: str) -> int:
        """Delete all chunks for the given source file.

        Args:
            source_file_id: ID of the source file.

        Returns:
            Number of chunks deleted.
        """
        col = self._get_collection()
        try:
            existing = col.get(where={"source_file_id": source_file_id})
            ids_to_delete = existing["ids"]
            if ids_to_delete:
                col.delete(ids=ids_to_delete)
            return len(ids_to_delete)
        except Exception as exc:
            raise VectorStoreError(f"ChromaDB delete failed: {exc}") from exc

    def get_stats(self) -> IndexStats:
        """Return collection statistics."""
        col = self._get_collection()
        try:
            count = col.count()
            all_meta = col.get(include=["metadatas"])["metadatas"]
        except Exception as exc:
            raise VectorStoreError(f"ChromaDB stats failed: {exc}") from exc

        languages: dict[str, int] = {}
        chunk_types: dict[str, int] = {}
        total_tokens = 0
        file_ids: set[str] = set()

        for meta in all_meta:
            lang = meta.get("language", "unknown")
            languages[lang] = languages.get(lang, 0) + 1
            ct = meta.get("chunk_type", "snippet")
            chunk_types[ct] = chunk_types.get(ct, 0) + 1
            total_tokens += meta.get("token_count", 0)
            file_ids.add(meta.get("source_file_id", ""))

        return IndexStats(
            total_files=len(file_ids),
            total_chunks=count,
            total_tokens=total_tokens,
            languages=languages,
            chunk_types=chunk_types,
            vector_store="chroma",
        )

    def clear(self) -> None:
        """Delete and recreate the ChromaDB collection."""
        client = self._get_client()
        try:
            client.delete_collection(self.collection_name)
            self._collection = None
            logger.info("ChromaDB collection %r cleared", self.collection_name)
        except Exception as exc:
            raise VectorStoreError(f"ChromaDB clear failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            import chromadb  # noqa: PLC0415

            if self._persist_directory:
                self._client = chromadb.PersistentClient(path=self._persist_directory)
            else:
                self._client = chromadb.EphemeralClient()
            return self._client
        except ImportError as exc:
            raise VectorStoreError(
                "chromadb not installed. Run: pip install chromadb"
            ) from exc

    def _get_collection(self) -> Any:
        if self._collection is not None:
            return self._collection
        client = self._get_client()
        metric = self._METRIC_MAP.get(self.distance_metric, "cosine")
        self._collection = client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": metric},
        )
        return self._collection

    @staticmethod
    def _chunk_to_meta(chunk: Chunk, model: str) -> dict[str, Any]:
        """Flatten a ``Chunk`` into ChromaDB-compatible metadata (primitives only)."""
        return {
            "source_file_id": chunk.source_file_id,
            "path": chunk.path,
            "language": chunk.language.value,
            "chunk_type": chunk.chunk_type.value,
            "name": chunk.name,
            "start_line": chunk.start_line,
            "end_line": chunk.end_line,
            "start_byte": chunk.start_byte,
            "end_byte": chunk.end_byte,
            "token_count": chunk.token_count,
            "context_before": chunk.context_before,
            "context_after": chunk.context_after,
            "embedding_model": model,
            # Serialise extra metadata as JSON string.
            "extra_metadata": json.dumps(chunk.metadata),
        }

    @staticmethod
    def _meta_to_chunk(chunk_id: str, meta: dict[str, Any], document: str) -> Chunk:
        """Reconstruct a ``Chunk`` from ChromaDB metadata + document text."""
        extra = {}
        if raw := meta.get("extra_metadata"):
            try:
                extra = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                pass

        return Chunk(
            id=chunk_id,
            source_file_id=meta.get("source_file_id", ""),
            path=meta.get("path", ""),
            language=Language(meta.get("language", "unknown")),
            chunk_type=ChunkType(meta.get("chunk_type", "snippet")),
            name=meta.get("name", ""),
            content=document,
            start_line=meta.get("start_line", 0),
            end_line=meta.get("end_line", 0),
            start_byte=meta.get("start_byte", 0),
            end_byte=meta.get("end_byte", 0),
            token_count=meta.get("token_count", 0),
            context_before=meta.get("context_before", ""),
            context_after=meta.get("context_after", ""),
            metadata=extra,
        )
