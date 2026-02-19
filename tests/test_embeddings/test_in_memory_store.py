"""Tests for the InMemoryVectorStore."""

from __future__ import annotations

import pytest

from code_indexer.core.models import Chunk, EmbeddedChunk, Language
from code_indexer.vectorstore.in_memory_store import InMemoryVectorStore, _dot, _l2_norm


# ---------------------------------------------------------------------------
# Unit tests for math helpers
# ---------------------------------------------------------------------------


def test_l2_norm_unit_vector() -> None:
    """Normalised vector should have magnitude ≈ 1."""
    import math
    v = [3.0, 4.0]
    n = _l2_norm(v)
    assert abs(math.sqrt(sum(x * x for x in n)) - 1.0) < 1e-9


def test_l2_norm_zero_vector_is_safe() -> None:
    v = [0.0, 0.0, 0.0]
    assert _l2_norm(v) == v  # no division by zero


def test_dot_product() -> None:
    assert _dot([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert _dot([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Store integration tests
# ---------------------------------------------------------------------------


def _make_chunk(chunk_id: str, content: str) -> Chunk:
    return Chunk(
        id=chunk_id,
        source_file_id="file1",
        path="src/test.py",
        language=Language.PYTHON,
        content=content,
        start_line=1,
        end_line=5,
    )


def _make_embedded(chunk: Chunk, vec: list[float]) -> EmbeddedChunk:
    return EmbeddedChunk(chunk=chunk, embedding=vec, model="test-model")


def test_upsert_and_query_returns_correct_result() -> None:
    store = InMemoryVectorStore(dimensions=2)
    chunk = _make_chunk("c1", "def authenticate(token): pass")
    ec = _make_embedded(chunk, [1.0, 0.0])
    store.upsert([ec])

    results = store.query([1.0, 0.0], top_k=1)
    assert len(results) == 1
    assert results[0].chunk.id == "c1"
    assert results[0].score == pytest.approx(1.0, abs=1e-5)


def test_upsert_replaces_existing_chunk() -> None:
    store = InMemoryVectorStore(dimensions=2)
    chunk = _make_chunk("c1", "original")
    ec = _make_embedded(chunk, [1.0, 0.0])
    store.upsert([ec])

    updated_chunk = _make_chunk("c1", "updated")
    store.upsert([_make_embedded(updated_chunk, [0.0, 1.0])])

    assert len(store._store) == 1
    assert store._store["c1"][1].chunk.content == "updated"


def test_delete_by_file_removes_all_chunks() -> None:
    store = InMemoryVectorStore(dimensions=2)
    chunks = [_make_chunk(f"c{i}", f"code {i}") for i in range(5)]
    for chunk in chunks:
        chunk.source_file_id = "file_to_delete"
    embedded = [_make_embedded(c, [1.0, 0.0]) for c in chunks]
    store.upsert(embedded)

    # Add a chunk from a different file.
    other_chunk = _make_chunk("other", "other code")
    other_chunk.source_file_id = "keep_file"
    store.upsert([_make_embedded(other_chunk, [1.0, 0.0])])

    deleted = store.delete_by_file("file_to_delete")
    assert deleted == 5
    assert "other" in store._store


def test_query_on_empty_store_returns_empty() -> None:
    store = InMemoryVectorStore(dimensions=2)
    results = store.query([1.0, 0.0], top_k=5)
    assert results == []


def test_get_stats_reflects_stored_chunks() -> None:
    store = InMemoryVectorStore(dimensions=2)
    chunks = [_make_chunk(f"c{i}", f"code {i}") for i in range(3)]
    embedded = [_make_embedded(c, [1.0, 0.0]) for c in chunks]
    store.upsert(embedded)

    stats = store.get_stats()
    assert stats.total_chunks == 3
    assert "python" in stats.languages


def test_clear_removes_all_data() -> None:
    store = InMemoryVectorStore(dimensions=2)
    chunk = _make_chunk("c1", "some code")
    store.upsert([_make_embedded(chunk, [1.0, 0.0])])
    store.clear()
    assert len(store._store) == 0
