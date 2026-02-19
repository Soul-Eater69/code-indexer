"""
Qdrant vector store backend.

Qdrant (https://qdrant.tech) is a production-grade, Rust-written vector
database that supports HNSW indexing, filtering, and horizontal scaling.

It can run as:
  * An in-process embedded database (same process, no server needed):
    ``QdrantVectorStore(url=":memory:")``
  * A Docker container: ``QdrantVectorStore(url="http://localhost:6333")``
  * Qdrant Cloud: ``QdrantVectorStore(url="https://xxx.qdrant.io", api_key="…")``

Payload schema
--------------
Each vector point stored in Qdrant has a ``payload`` dict containing all
chunk metadata.  We store the full chunk content there too so we can
reconstruct ``Chunk`` objects from query results without a separate lookup.
"""

from __future__ import annotations

import json
import logging
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


class QdrantVectorStore(BaseVectorStore):
    """Store and retrieve code embeddings using Qdrant.

    Args:
        collection_name:  Qdrant collection name.
        url:              Qdrant server URL, or ``:memory:`` for embedded mode.
        api_key:          Optional API key for Qdrant Cloud.
        dimensions:       Embedding vector dimensionality.
        distance_metric:  ``"cosine"``, ``"dot"``, or ``"euclidean"``.
    """

    _METRIC_MAP = {
        "cosine": "Cosine",
        "dot": "Dot",
        "euclidean": "Euclid",
    }

    def __init__(
        self,
        collection_name: str = "code_index",
        url: str = "http://localhost:6333",
        api_key: str | None = None,
        dimensions: int = 384,
        distance_metric: str = "cosine",
    ) -> None:
        super().__init__(
            collection_name=collection_name,
            dimensions=dimensions,
            distance_metric=distance_metric,
        )
        self._url = url
        self._api_key = api_key
        self._client: Any = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def upsert(self, embedded_chunks: list[EmbeddedChunk]) -> int:
        """Upsert *embedded_chunks* into Qdrant.

        Args:
            embedded_chunks: Chunks with vectors.

        Returns:
            Number of chunks stored.
        """
        if not embedded_chunks:
            return 0

        from qdrant_client.models import PointStruct  # noqa: PLC0415

        client = self._get_client()
        self._ensure_collection(client)

        points = [
            PointStruct(
                id=self._chunk_id_to_int(ec.chunk.id),
                vector=ec.embedding,
                payload=self._chunk_to_payload(ec.chunk, ec.model),
            )
            for ec in embedded_chunks
        ]

        try:
            client.upsert(collection_name=self.collection_name, points=points)
            logger.debug("Qdrant upserted %d points", len(points))
            return len(points)
        except Exception as exc:
            raise VectorStoreError(f"Qdrant upsert failed: {exc}") from exc

    def query(
        self,
        query_vector: list[float],
        top_k: int = 10,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        """Query Qdrant for nearest neighbours.

        Args:
            query_vector: Dense query embedding.
            top_k:        Max results.
            filters:      Dict of payload field → value for exact filtering.
                          Translated to Qdrant ``Filter`` objects internally.

        Returns:
            List of ``SearchResult`` sorted by descending score.
        """
        client = self._get_client()
        qdrant_filter = self._build_filter(filters) if filters else None

        try:
            results = client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
                query_filter=qdrant_filter,
                with_payload=True,
            )
        except Exception as exc:
            raise VectorStoreError(f"Qdrant query failed: {exc}") from exc

        search_results: list[SearchResult] = []
        for rank, hit in enumerate(results, start=1):
            chunk = self._payload_to_chunk(hit.id, hit.payload or {})
            search_results.append(
                SearchResult(chunk=chunk, score=float(hit.score), rank=rank)
            )
        return search_results

    def delete_by_file(self, source_file_id: str) -> int:
        """Delete all chunks for *source_file_id* from Qdrant.

        Args:
            source_file_id: Source file ID to purge.

        Returns:
            Number of points deleted (approximate for Qdrant).
        """
        from qdrant_client.models import FieldCondition, Filter, MatchValue  # noqa: PLC0415

        client = self._get_client()
        try:
            client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="source_file_id",
                            match=MatchValue(value=source_file_id),
                        )
                    ]
                ),
            )
            # Qdrant delete doesn't return count; return -1 as sentinel.
            return -1
        except Exception as exc:
            raise VectorStoreError(f"Qdrant delete failed: {exc}") from exc

    def get_stats(self) -> IndexStats:
        """Return collection statistics from Qdrant."""
        client = self._get_client()
        try:
            info = client.get_collection(self.collection_name)
            count = info.points_count or 0
        except Exception as exc:
            raise VectorStoreError(f"Qdrant stats failed: {exc}") from exc

        return IndexStats(
            total_chunks=count,
            vector_store="qdrant",
        )

    def clear(self) -> None:
        """Delete and recreate the Qdrant collection."""
        client = self._get_client()
        try:
            client.delete_collection(self.collection_name)
            self._ensure_collection(client)
            logger.info("Qdrant collection %r cleared", self.collection_name)
        except Exception as exc:
            raise VectorStoreError(f"Qdrant clear failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from qdrant_client import QdrantClient  # noqa: PLC0415

            kwargs: dict = {"url": self._url}
            if self._api_key:
                kwargs["api_key"] = self._api_key
            if self._url == ":memory:":
                self._client = QdrantClient(":memory:")
            else:
                self._client = QdrantClient(**kwargs)
            return self._client
        except ImportError as exc:
            raise VectorStoreError(
                "qdrant-client not installed. Run: pip install qdrant-client"
            ) from exc

    def _ensure_collection(self, client: Any) -> None:
        """Create the collection if it does not yet exist."""
        from qdrant_client.models import Distance, VectorParams  # noqa: PLC0415

        distance_str = self._METRIC_MAP.get(self.distance_metric, "Cosine")
        distance = getattr(Distance, distance_str.upper(), Distance.COSINE)

        existing = [c.name for c in client.get_collections().collections]
        if self.collection_name not in existing:
            client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.dimensions, distance=distance),
            )
            logger.info("Created Qdrant collection %r", self.collection_name)

    @staticmethod
    def _chunk_id_to_int(chunk_id: str) -> int:
        """Convert a hex string chunk ID to an integer (Qdrant requires int IDs)."""
        # Take first 15 hex chars → 60-bit integer (well within int64 range).
        return int(chunk_id[:15], 16)

    @staticmethod
    def _chunk_to_payload(chunk: Chunk, model: str) -> dict[str, Any]:
        """Serialise a ``Chunk`` to a Qdrant payload dict."""
        return {
            "chunk_id": chunk.id,
            "source_file_id": chunk.source_file_id,
            "path": chunk.path,
            "language": chunk.language.value,
            "chunk_type": chunk.chunk_type.value,
            "name": chunk.name,
            "content": chunk.content,
            "start_line": chunk.start_line,
            "end_line": chunk.end_line,
            "start_byte": chunk.start_byte,
            "end_byte": chunk.end_byte,
            "token_count": chunk.token_count,
            "context_before": chunk.context_before,
            "context_after": chunk.context_after,
            "embedding_model": model,
            "extra_metadata": json.dumps(chunk.metadata),
        }

    @staticmethod
    def _payload_to_chunk(point_id: int | str, payload: dict[str, Any]) -> Chunk:
        """Reconstruct a ``Chunk`` from a Qdrant payload."""
        extra = {}
        if raw := payload.get("extra_metadata"):
            try:
                extra = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                pass

        return Chunk(
            id=payload.get("chunk_id", str(point_id)),
            source_file_id=payload.get("source_file_id", ""),
            path=payload.get("path", ""),
            language=Language(payload.get("language", "unknown")),
            chunk_type=ChunkType(payload.get("chunk_type", "snippet")),
            name=payload.get("name", ""),
            content=payload.get("content", ""),
            start_line=payload.get("start_line", 0),
            end_line=payload.get("end_line", 0),
            start_byte=payload.get("start_byte", 0),
            end_byte=payload.get("end_byte", 0),
            token_count=payload.get("token_count", 0),
            context_before=payload.get("context_before", ""),
            context_after=payload.get("context_after", ""),
            metadata=extra,
        )

    @staticmethod
    def _build_filter(filters: dict) -> Any:
        """Convert a flat filters dict to a Qdrant ``Filter`` object."""
        from qdrant_client.models import FieldCondition, Filter, MatchValue  # noqa: PLC0415

        conditions = [
            FieldCondition(key=k, match=MatchValue(value=v))
            for k, v in filters.items()
        ]
        return Filter(must=conditions)
