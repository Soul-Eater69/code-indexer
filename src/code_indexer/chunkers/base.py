"""
Abstract base class for all chunking strategies.

A chunker's sole job is to turn a ``ParsedFile`` into a list of ``Chunk``
objects.  Chunkers are stateless: all configuration is passed at construction
time and ``chunk`` can be called concurrently from multiple threads.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from code_indexer.core.models import Chunk, ParsedFile


class BaseChunker(ABC):
    """Abstract chunker interface.

    Args:
        chunk_size:    Target size in the unit appropriate for this chunker
                       (tokens for token-based, characters for char-based).
        chunk_overlap: How many units to overlap between consecutive chunks.
        min_chunk_size: Chunks smaller than this are discarded.
        max_chunk_size: Chunks larger than this are split further.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        min_chunk_size: int = 20,
        max_chunk_size: int = 2048,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size

    @abstractmethod
    def chunk(self, parsed_file: ParsedFile) -> list[Chunk]:
        """Split *parsed_file* into a list of ``Chunk`` objects.

        Implementations must:
        * Never raise for valid inputs (return an empty list instead).
        * Produce non-overlapping or minimally-overlapping chunks.
        * Populate at least: ``source_file_id``, ``path``, ``language``,
          ``content``, ``start_line``, ``end_line``, ``start_byte``,
          ``end_byte``.

        Args:
            parsed_file: Output of a parser run.

        Returns:
            Ordered list of chunks (file order preserved).
        """

    @property
    def name(self) -> str:
        """Human-readable name of this chunker."""
        return type(self).__name__

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _build_chunk(
        self,
        parsed_file: ParsedFile,
        content: str,
        start_line: int,
        end_line: int,
        start_byte: int,
        end_byte: int,
        **kwargs: object,
    ) -> Chunk | None:
        """Factory helper: build a ``Chunk`` and apply size guards.

        Returns ``None`` when the content is smaller than ``min_chunk_size``
        so callers can easily filter with ``filter(None, ...)`` patterns.
        """
        if len(content.split()) < self.min_chunk_size:
            return None  # too small to be useful

        source = parsed_file.source
        return Chunk(
            source_file_id=source.id,
            path=source.path,
            language=source.language,
            content=content,
            start_line=start_line,
            end_line=end_line,
            start_byte=start_byte,
            end_byte=end_byte,
            metadata={**source.metadata},
            **kwargs,  # type: ignore[arg-type]
        )

    def _add_context(
        self,
        chunk: Chunk,
        all_lines: list[str],
        context_lines: int = 3,
    ) -> None:
        """Populate ``context_before`` / ``context_after`` on *chunk* in-place.

        Args:
            chunk:         The chunk to annotate.
            all_lines:     All lines of the source file (0-indexed).
            context_lines: Number of lines before/after to capture.
        """
        before_start = max(0, chunk.start_line - 1 - context_lines)
        before_end = chunk.start_line - 1
        chunk.context_before = "\n".join(all_lines[before_start:before_end])

        after_start = chunk.end_line
        after_end = min(len(all_lines), chunk.end_line + context_lines)
        chunk.context_after = "\n".join(all_lines[after_start:after_end])
