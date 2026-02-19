"""
Token-count-aware chunker.

What problem does this solve?
------------------------------
LLMs have context-window limits measured in *tokens*, not characters.  A chunk
that looks small in characters might be large in tokens because of dense code
(e.g. minified JS).  This chunker measures chunk size in tokens so you can
precisely control how many tokens each chunk occupies.

Tokenizer choice
-----------------
We default to ``tiktoken`` (OpenAI's tokenizer, ``cl100k_base`` encoding)
because it is fast, has a small footprint, and correlates well with most
modern LLMs.  Any ``tokenizers``-compatible tokenizer can be plugged in by
subclassing and overriding ``_count_tokens`` and ``_split_tokens``.

Algorithm
----------
1. Split the source content into lines.
2. Accumulate lines into a buffer, counting tokens as we go.
3. When the buffer would exceed ``chunk_size``, flush it as a chunk.
4. Apply ``chunk_overlap``: the next chunk starts by re-adding the last
   ``chunk_overlap`` tokens of the previous chunk.
5. Repeat until all lines are processed.
6. Flush any remaining buffered lines as the final chunk.

Token counting note
--------------------
Token counting is O(n) in the number of tokens and runs in Python.  For very
large files (>10k lines) this can be slow.  As an optimisation we count tokens
on batches of lines rather than character-by-character.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from code_indexer.chunkers.base import BaseChunker
from code_indexer.core.models import Chunk, ChunkType, ParsedFile

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_encoding() -> object:
    """Return a cached tiktoken encoding object.

    We cache at module level to avoid re-loading the BPE vocabulary on every
    chunk call (loading ``cl100k_base`` takes ~50ms on first access).
    """
    try:
        import tiktoken  # noqa: PLC0415

        return tiktoken.get_encoding("cl100k_base")
    except ImportError:
        logger.warning(
            "tiktoken not installed; falling back to whitespace token counting"
        )
        return None


class TokenChunker(BaseChunker):
    """Split source code into chunks bounded by token count.

    Args:
        chunk_size:    Maximum tokens per chunk.
        chunk_overlap: Number of tokens to repeat at the start of each new
                       chunk for context continuity.
        min_chunk_size: Minimum *word* count (not tokens) for a chunk to be
                        kept.  Small leftover fragments are discarded.
        max_chunk_size: Hard upper bound in tokens.  Chunks exceeding this
                        are forcibly split at line boundaries.
        encoding_name:  tiktoken encoding name.  ``cl100k_base`` covers GPT-4
                        and later; use ``p50k_base`` for GPT-3.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        min_chunk_size: int = 20,
        max_chunk_size: int = 2048,
        encoding_name: str = "cl100k_base",
    ) -> None:
        super().__init__(chunk_size, chunk_overlap, min_chunk_size, max_chunk_size)
        self.encoding_name = encoding_name

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk(self, parsed_file: ParsedFile) -> list[Chunk]:
        """Split *parsed_file* into token-bounded chunks.

        Args:
            parsed_file: Parsed source file.

        Returns:
            List of ``Chunk`` objects.
        """
        source = parsed_file.source
        lines = source.content.splitlines(keepends=True)
        if not lines:
            return []

        chunks: list[Chunk] = []
        buffer_lines: list[str] = []
        buffer_tokens: int = 0
        buffer_start_line: int = 1  # 1-indexed
        byte_offset: int = 0
        chunk_start_byte: int = 0

        # We also track the last chunk's trailing text for overlap.
        overlap_text: str = ""

        for line_idx, line in enumerate(lines, start=1):
            line_tokens = self._count_tokens(line)

            # Would adding this line exceed our chunk size?
            if buffer_tokens + line_tokens > self.chunk_size and buffer_lines:
                # Flush the current buffer as a chunk.
                chunk_text = overlap_text + "".join(buffer_lines)
                chunk_end_byte = byte_offset
                chunk = self._build_chunk(
                    parsed_file=parsed_file,
                    content=chunk_text,
                    start_line=buffer_start_line,
                    end_line=line_idx - 1,
                    start_byte=chunk_start_byte,
                    end_byte=chunk_end_byte,
                    chunk_type=ChunkType.SNIPPET,
                )
                if chunk is not None:
                    chunks.append(chunk)

                # Prepare overlap: take the last ``chunk_overlap`` tokens worth
                # of text from the buffer to prepend to the next chunk.
                overlap_text = self._get_overlap_text(
                    "".join(buffer_lines), self.chunk_overlap
                )

                buffer_lines = []
                buffer_tokens = 0
                buffer_start_line = line_idx
                chunk_start_byte = byte_offset

            buffer_lines.append(line)
            buffer_tokens += line_tokens
            byte_offset += len(line.encode("utf-8", errors="replace"))

        # Flush remaining lines.
        if buffer_lines:
            chunk_text = overlap_text + "".join(buffer_lines)
            chunk = self._build_chunk(
                parsed_file=parsed_file,
                content=chunk_text,
                start_line=buffer_start_line,
                end_line=len(lines),
                start_byte=chunk_start_byte,
                end_byte=byte_offset,
                chunk_type=ChunkType.SNIPPET,
            )
            if chunk is not None:
                chunks.append(chunk)

        logger.debug(
            "TokenChunker: %s → %d chunks",
            source.path,
            len(chunks),
        )
        return chunks

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _count_tokens(self, text: str) -> int:
        """Count tokens in *text* using tiktoken or whitespace fallback."""
        enc = _get_encoding()
        if enc is None:
            # Whitespace approximation: ~1.3 chars per token for code.
            return max(1, len(text) // 4)
        return len(enc.encode(text, disallowed_special=()))  # type: ignore[union-attr]

    def _get_overlap_text(self, text: str, overlap_tokens: int) -> str:
        """Return the last *overlap_tokens* tokens of *text* as a string."""
        if overlap_tokens == 0:
            return ""
        enc = _get_encoding()
        if enc is None:
            # Approximate: 4 chars per token.
            char_limit = overlap_tokens * 4
            return text[-char_limit:] if len(text) > char_limit else text
        tokens = enc.encode(text, disallowed_special=())  # type: ignore[union-attr]
        tail_tokens = tokens[-overlap_tokens:]
        return enc.decode(tail_tokens)  # type: ignore[union-attr]
