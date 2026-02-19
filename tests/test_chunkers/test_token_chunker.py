"""Tests for the token-based chunker."""

from __future__ import annotations

import pytest

from code_indexer.chunkers.token_chunker import TokenChunker
from code_indexer.core.models import Language, ParsedFile, SourceFile


def _make_parsed(content: str, language: Language = Language.PYTHON) -> ParsedFile:
    source = SourceFile(path="test.py", content=content, language=language)
    return ParsedFile(source=source)


def test_empty_file_returns_no_chunks() -> None:
    chunker = TokenChunker(min_chunk_size=1)
    result = chunker.chunk(_make_parsed(""))
    assert result == []


def test_small_file_produces_one_chunk() -> None:
    chunker = TokenChunker(chunk_size=512, min_chunk_size=1)
    content = "def hello():\n    return 'world'\n"
    chunks = chunker.chunk(_make_parsed(content))
    assert len(chunks) == 1
    assert "hello" in chunks[0].content


def test_chunks_have_correct_language() -> None:
    chunker = TokenChunker(chunk_size=512, min_chunk_size=1)
    chunks = chunker.chunk(_make_parsed("x = 1\n", Language.PYTHON))
    for chunk in chunks:
        assert chunk.language == Language.PYTHON


def test_chunk_fields_populated() -> None:
    chunker = TokenChunker(chunk_size=512, min_chunk_size=1)
    content = "a = 1\nb = 2\nc = 3\n"
    chunks = chunker.chunk(_make_parsed(content))
    assert len(chunks) >= 1
    chunk = chunks[0]
    assert chunk.source_file_id != ""
    assert chunk.path == "test.py"
    assert chunk.start_line >= 1
    assert chunk.end_line >= chunk.start_line
    assert chunk.content.strip() != ""


def test_large_file_splits_into_multiple_chunks() -> None:
    # Generate a file large enough to require splitting at chunk_size=50.
    lines = [f"variable_{i} = {i}  # some comment here\n" for i in range(200)]
    content = "".join(lines)
    chunker = TokenChunker(chunk_size=50, chunk_overlap=5, min_chunk_size=1)
    chunks = chunker.chunk(_make_parsed(content))
    assert len(chunks) > 1


def test_chunk_ids_are_unique() -> None:
    lines = [f"x_{i} = {i}\n" for i in range(200)]
    content = "".join(lines)
    chunker = TokenChunker(chunk_size=50, chunk_overlap=0, min_chunk_size=1)
    chunks = chunker.chunk(_make_parsed(content))
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))
