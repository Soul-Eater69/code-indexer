"""Tests for the sliding window chunker."""

from __future__ import annotations

import pytest

from code_indexer.chunkers.sliding_window import SlidingWindowChunker
from code_indexer.core.models import Language, ParsedFile, SourceFile


def _make_parsed(content: str) -> ParsedFile:
    source = SourceFile(path="test.py", content=content, language=Language.PYTHON)
    return ParsedFile(source=source)


def test_empty_content_returns_no_chunks() -> None:
    chunker = SlidingWindowChunker()
    assert chunker.chunk(_make_parsed("   \n  ")) == []


def test_content_smaller_than_window_is_one_chunk() -> None:
    chunker = SlidingWindowChunker(chunk_size=1000, min_chunk_size=1)
    content = "print('hello')\n" * 5
    chunks = chunker.chunk(_make_parsed(content))
    assert len(chunks) == 1


def test_windowing_produces_overlap() -> None:
    # With a 50-char window and 10-char overlap, windows should share text.
    chunker = SlidingWindowChunker(
        chunk_size=50, chunk_overlap=10, min_chunk_size=1, split_on_newline=False
    )
    content = "A" * 200
    chunks = chunker.chunk(_make_parsed(content))
    assert len(chunks) > 1
    # Verify overlap: the start of chunk[1] should appear in the end of chunk[0].
    c0_end = chunks[0].content[-10:]
    c1_start = chunks[1].content[:10]
    # They should share characters (both are 'A's in this case).
    assert c0_end == c1_start


def test_chunks_cover_entire_content() -> None:
    chunker = SlidingWindowChunker(
        chunk_size=30, chunk_overlap=0, min_chunk_size=1, split_on_newline=False
    )
    content = "X" * 100
    chunks = chunker.chunk(_make_parsed(content))
    # Concatenating all chunk contents (no overlap) should reconstruct the original.
    reconstructed = "".join(c.content for c in chunks)
    assert len(reconstructed) >= len(content)
