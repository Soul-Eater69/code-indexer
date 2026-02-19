"""
Main indexing pipeline.

The pipeline orchestrates all stages:

  ┌──────────────┐
  │ CodebaseWalker│  Yield SourceFile objects
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐
  │ TreeSitterParser│  Parse → ParsedFile (AST nodes)
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐
  │   Chunker    │  Chunk → list[Chunk]
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐
  │   Embedder   │  Embed → list[EmbeddedChunk]
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐
  │  VectorStore │  Upsert / persist
  └──────────────┘

Each stage is optional: you can skip the embedder to produce a pure
text index, or skip the vector store to just collect chunks for inspection.

Concurrency
-----------
File I/O and parsing are CPU-bound in Python (GIL).  We keep the pipeline
single-threaded here for simplicity and correctness.  For large codebases,
consider splitting the walker across multiple processes and merging the
resulting chunks before embedding (embedding I/O is the real bottleneck).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from code_indexer.chunkers.ast_chunker import ASTChunker
from code_indexer.chunkers.base import BaseChunker
from code_indexer.chunkers.sliding_window import SlidingWindowChunker
from code_indexer.chunkers.token_chunker import TokenChunker
from code_indexer.core.config import Settings, get_settings
from code_indexer.core.models import Chunk, EmbeddedChunk, IndexStats, SourceFile
from code_indexer.embeddings.base import BaseEmbedder
from code_indexer.indexer.walker import CodebaseWalker
from code_indexer.parsers.tree_sitter_parser import TreeSitterParser
from code_indexer.vectorstore.base import BaseVectorStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Progress callback type
# ---------------------------------------------------------------------------

ProgressCallback = Callable[[str, int, int], None]
"""
Signature: (message: str, current: int, total: int) -> None

Called by the pipeline at key milestones so callers can update progress bars
or logging without coupling to a specific UI framework.
"""


# ---------------------------------------------------------------------------
# Pipeline result
# ---------------------------------------------------------------------------


@dataclass
class IndexResult:
    """Summary of a completed indexing run.

    Attributes:
        files_processed:  Number of source files successfully processed.
        files_skipped:    Number of files skipped (binary, too large, etc.).
        chunks_produced:  Total chunks extracted from all files.
        chunks_embedded:  Chunks that were successfully embedded.
        chunks_stored:    Chunks successfully written to the vector store.
        elapsed_seconds:  Wall-clock time for the entire pipeline.
        errors:           List of (path, error_message) for any failures.
    """

    files_processed: int = 0
    files_skipped: int = 0
    chunks_produced: int = 0
    chunks_embedded: int = 0
    chunks_stored: int = 0
    elapsed_seconds: float = 0.0
    errors: list[tuple[str, str]] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Fraction of files that were indexed without errors."""
        total = self.files_processed + len(self.errors)
        return self.files_processed / total if total else 0.0


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


def build_chunker(settings: Settings) -> BaseChunker:
    """Instantiate the configured chunker from *settings*.

    Args:
        settings: Application settings.

    Returns:
        A ``BaseChunker`` subclass instance.
    """
    s = settings.chunker
    match s.strategy:
        case "ast":
            return ASTChunker(
                chunk_size=s.chunk_size,
                chunk_overlap=s.chunk_overlap,
                min_chunk_size=s.min_chunk_size,
                max_chunk_size=s.max_chunk_size,
            )
        case "token":
            return TokenChunker(
                chunk_size=s.chunk_size,
                chunk_overlap=s.chunk_overlap,
                min_chunk_size=s.min_chunk_size,
                max_chunk_size=s.max_chunk_size,
            )
        case "sliding_window":
            return SlidingWindowChunker(
                chunk_size=s.chunk_size,
                chunk_overlap=s.chunk_overlap,
                min_chunk_size=s.min_chunk_size,
            )
        case _:
            logger.warning("Unknown strategy %r, defaulting to ASTChunker", s.strategy)
            return ASTChunker(chunk_size=s.chunk_size)


def build_embedder(settings: Settings) -> BaseEmbedder | None:
    """Instantiate the configured embedder from *settings*.

    Returns ``None`` if embedding is disabled (no provider configured).

    Args:
        settings: Application settings.

    Returns:
        A ``BaseEmbedder`` subclass instance, or ``None``.
    """
    s = settings.embedding
    match s.provider:
        case "openai":
            from code_indexer.embeddings.openai_embedder import OpenAIEmbedder  # noqa: PLC0415

            return OpenAIEmbedder(
                model=s.model,
                api_key=s.openai_api_key,
                batch_size=s.batch_size,
                dimensions=s.dimensions,
            )
        case "sentence_transformer":
            from code_indexer.embeddings.sentence_transformer_embedder import (  # noqa: PLC0415
                SentenceTransformerEmbedder,
            )

            return SentenceTransformerEmbedder(
                model=s.model,
                batch_size=s.batch_size,
            )
        case "ollama":
            from code_indexer.embeddings.ollama_embedder import OllamaEmbedder  # noqa: PLC0415

            return OllamaEmbedder(
                model=s.model,
                base_url=s.ollama_base_url,
                batch_size=s.batch_size,
                dimensions=s.dimensions,
            )
        case _:
            logger.warning("Unknown embedding provider %r, skipping", s.provider)
            return None


def build_vector_store(settings: Settings) -> BaseVectorStore | None:
    """Instantiate the configured vector store from *settings*.

    Returns ``None`` when the provider is unknown or not configured.

    Args:
        settings: Application settings.

    Returns:
        A ``BaseVectorStore`` subclass instance, or ``None``.
    """
    s = settings.vectorstore
    match s.provider:
        case "chroma":
            from code_indexer.vectorstore.chroma_store import ChromaVectorStore  # noqa: PLC0415

            return ChromaVectorStore(
                collection_name=s.collection_name,
                persist_directory=s.chroma_persist_dir,
                distance_metric=s.distance_metric,
            )
        case "qdrant":
            from code_indexer.vectorstore.qdrant_store import QdrantVectorStore  # noqa: PLC0415

            return QdrantVectorStore(
                collection_name=s.collection_name,
                url=s.qdrant_url,
                api_key=s.qdrant_api_key or None,
                distance_metric=s.distance_metric,
            )
        case "in_memory":
            from code_indexer.vectorstore.in_memory_store import InMemoryVectorStore  # noqa: PLC0415

            return InMemoryVectorStore(
                collection_name=s.collection_name,
                distance_metric=s.distance_metric,
            )
        case _:
            logger.warning("Unknown vector store provider %r", s.provider)
            return None


# ---------------------------------------------------------------------------
# Main pipeline class
# ---------------------------------------------------------------------------


class IndexingPipeline:
    """Orchestrate the full codebase indexing pipeline.

    Args:
        settings:       Application settings.  Defaults to the global
                        singleton from ``get_settings()``.
        chunker:        Chunker to use.  Built from settings if ``None``.
        embedder:       Embedder backend.  Built from settings if ``None``.
        vector_store:   Vector store backend.  Built from settings if ``None``.
        progress_cb:    Optional callback for progress reporting.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        chunker: BaseChunker | None = None,
        embedder: BaseEmbedder | None = None,
        vector_store: BaseVectorStore | None = None,
        progress_cb: ProgressCallback | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._parser = TreeSitterParser()
        self._chunker = chunker or build_chunker(self._settings)
        self._embedder = embedder or build_embedder(self._settings)
        self._vector_store = vector_store or build_vector_store(self._settings)
        self._progress_cb = progress_cb or _noop_progress

        logger.info(
            "IndexingPipeline ready: chunker=%s embedder=%s store=%s",
            self._chunker.name,
            self._embedder.name if self._embedder else "None",
            self._vector_store.name if self._vector_store else "None",
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def index_directory(
        self,
        path: str | Path,
        extra_metadata: dict | None = None,
        clear_existing: bool = False,
    ) -> IndexResult:
        """Index an entire codebase directory.

        Args:
            path:           Path to the codebase root directory.
            extra_metadata: Extra metadata to attach to every ``SourceFile``.
            clear_existing: If ``True``, wipe the vector store before indexing.
                            Useful for a clean re-index.

        Returns:
            ``IndexResult`` with statistics about this run.
        """
        start = time.perf_counter()
        path = Path(path).resolve()
        result = IndexResult()

        if clear_existing and self._vector_store:
            logger.info("Clearing existing vector store collection …")
            self._vector_store.clear()

        walker = CodebaseWalker(
            root=path,
            ignore_patterns=self._settings.parser.ignore_patterns,
            max_file_size=self._settings.parser.max_file_size_bytes,
            extra_metadata=extra_metadata or {},
        )

        total_files = walker.count_files()
        logger.info("Found %d files to index in %s", total_files, path)
        self._progress_cb("Counting files", 0, total_files)

        all_chunks: list[Chunk] = []

        for file_idx, source_file in enumerate(walker.walk(), start=1):
            self._progress_cb(f"Parsing {source_file.path}", file_idx, total_files)
            try:
                file_chunks = self._process_file(source_file)
                all_chunks.extend(file_chunks)
                result.chunks_produced += len(file_chunks)
                result.files_processed += 1
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to process %s: %s", source_file.path, exc)
                result.errors.append((source_file.path, str(exc)))

        # Embed and store all chunks in one pass.
        if all_chunks and self._embedder:
            embedded = self._embed_chunks(all_chunks, result)
            if embedded and self._vector_store:
                stored = self._vector_store.upsert(embedded)
                result.chunks_stored = stored
                logger.info("Stored %d chunks in vector store", stored)

        result.elapsed_seconds = time.perf_counter() - start
        logger.info(
            "Indexing complete: %d files, %d chunks, %.1fs",
            result.files_processed,
            result.chunks_produced,
            result.elapsed_seconds,
        )
        return result

    def index_file(self, path: str | Path) -> IndexResult:
        """Index a single file.

        Args:
            path: Path to the source file.

        Returns:
            ``IndexResult`` with statistics.
        """
        from code_indexer.parsers.language_registry import detect_language  # noqa: PLC0415

        start = time.perf_counter()
        path = Path(path).resolve()
        result = IndexResult()

        try:
            raw_bytes = path.read_bytes()
            content = raw_bytes.decode("utf-8", errors="replace")
            source_file = SourceFile(
                path=str(path),
                content=content,
                language=detect_language(path),
                size_bytes=path.stat().st_size,
            )
            file_chunks = self._process_file(source_file)
            result.chunks_produced = len(file_chunks)
            result.files_processed = 1

            if file_chunks and self._embedder:
                embedded = self._embed_chunks(file_chunks, result)
                if embedded and self._vector_store:
                    result.chunks_stored = self._vector_store.upsert(embedded)

        except Exception as exc:
            result.errors.append((str(path), str(exc)))

        result.elapsed_seconds = time.perf_counter() - start
        return result

    def get_stats(self) -> IndexStats | None:
        """Return current index statistics from the vector store."""
        if self._vector_store is None:
            return None
        return self._vector_store.get_stats()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _process_file(self, source_file: SourceFile) -> list[Chunk]:
        """Parse and chunk a single source file.

        Args:
            source_file: The file to process.

        Returns:
            List of extracted chunks.
        """
        parsed = self._parser.parse(source_file)
        chunks = self._chunker.chunk(parsed)
        logger.debug("%s → %d chunks", source_file.path, len(chunks))
        return chunks

    def _embed_chunks(
        self, chunks: list[Chunk], result: IndexResult
    ) -> list[EmbeddedChunk]:
        """Embed all chunks and update *result* counters.

        Args:
            chunks: Chunks to embed.
            result: Result object to update in-place.

        Returns:
            List of ``EmbeddedChunk`` objects.
        """
        assert self._embedder is not None  # guarded by caller
        total = len(chunks)
        self._progress_cb("Embedding chunks", 0, total)

        try:
            embedded = self._embedder.embed_chunks(chunks)
            result.chunks_embedded = len(embedded)
            self._progress_cb("Embedding done", total, total)
            return embedded
        except Exception as exc:
            logger.error("Embedding failed: %s", exc)
            result.errors.append(("embedding", str(exc)))
            return []


def _noop_progress(message: str, current: int, total: int) -> None:
    """Default no-op progress callback."""
