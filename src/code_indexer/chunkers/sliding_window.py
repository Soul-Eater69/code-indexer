"""
Character-based sliding window chunker.

This is the simplest possible chunker and is provided as a baseline / fallback.
It makes no attempt to understand the code structure: it splits on character
count with a configurable overlap.

When to use it
--------------
* As a fallback when Tree-sitter grammars are unavailable.
* For non-code text files (markdown, YAML configs, …) where AST parsing is
  inappropriate.
* When you need deterministic, reproducible chunk boundaries independent of
  the tokenizer used at inference time.

Algorithm
----------
  index = 0
  while index < len(content):
      chunk = content[index : index + chunk_size]
      emit chunk
      index += (chunk_size - chunk_overlap)

The step ``chunk_size - chunk_overlap`` ensures each new window starts
``chunk_overlap`` characters before the end of the previous window.
"""

from __future__ import annotations

import logging

from code_indexer.chunkers.base import BaseChunker
from code_indexer.core.models import Chunk, ChunkType, ParsedFile

logger = logging.getLogger(__name__)


class SlidingWindowChunker(BaseChunker):
    """Character-based sliding window chunker.

    Args:
        chunk_size:    Maximum number of characters per chunk.
        chunk_overlap: Number of characters to repeat at the start of each
                       subsequent chunk.
        min_chunk_size: Minimum word count for a chunk to be kept.
        max_chunk_size: Not used by this chunker (each window is exactly
                        ``chunk_size`` characters or smaller).
        split_on_newline:
            When ``True`` (default), snap chunk boundaries to the nearest
            preceding newline.  This prevents splitting in the middle of a
            token / word.
    """

    def __init__(
        self,
        chunk_size: int = 2000,      # characters, not tokens
        chunk_overlap: int = 200,
        min_chunk_size: int = 20,
        max_chunk_size: int = 8000,
        split_on_newline: bool = True,
    ) -> None:
        super().__init__(chunk_size, chunk_overlap, min_chunk_size, max_chunk_size)
        self.split_on_newline = split_on_newline

    def chunk(self, parsed_file: ParsedFile) -> list[Chunk]:
        """Chunk *parsed_file* with a sliding character window.

        Args:
            parsed_file: Parsed source file.

        Returns:
            List of ``Chunk`` objects with non-overlapping main content.
        """
        source = parsed_file.source
        content = source.content
        if not content.strip():
            return []

        step = max(1, self.chunk_size - self.chunk_overlap)
        chunks: list[Chunk] = []
        index = 0
        raw_bytes = content.encode("utf-8", errors="replace")

        while index < len(content):
            end = min(index + self.chunk_size, len(content))

            # Snap to newline boundary to avoid mid-word splits.
            if self.split_on_newline and end < len(content):
                newline_pos = content.rfind("\n", index, end)
                if newline_pos > index:
                    end = newline_pos + 1  # include the newline in this chunk

            window_text = content[index:end]

            # Compute line numbers from character positions.
            start_line = content[:index].count("\n") + 1
            end_line = content[:end].count("\n") + 1

            # Byte offsets for the window.
            start_byte = len(content[:index].encode("utf-8", errors="replace"))
            end_byte = len(content[:end].encode("utf-8", errors="replace"))

            chunk = self._build_chunk(
                parsed_file=parsed_file,
                content=window_text,
                start_line=start_line,
                end_line=end_line,
                start_byte=start_byte,
                end_byte=end_byte,
                chunk_type=ChunkType.SNIPPET,
            )
            if chunk is not None:
                chunks.append(chunk)

            if end >= len(content):
                break
            index += step

        logger.debug(
            "SlidingWindowChunker: %s → %d chunks",
            source.path,
            len(chunks),
        )
        return chunks
