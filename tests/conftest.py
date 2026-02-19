"""
Shared pytest fixtures.

These fixtures are automatically available in all test files without
explicit imports — pytest discovers them from conftest.py automatically.

Fixture scope guide:
  * ``session`` – created once per test session (expensive resources).
  * ``module``  – created once per test module file.
  * ``function`` – default; created fresh for every test (isolated, safe).
"""

from __future__ import annotations

import pytest

from code_indexer.core.config import Settings, get_settings
from code_indexer.core.models import Language, ParsedFile, SourceFile
from code_indexer.vectorstore.in_memory_store import InMemoryVectorStore


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def test_settings() -> Settings:
    """Return a minimal Settings instance for tests.

    We override the vector store to in_memory so no ChromaDB or Qdrant
    server is needed during CI.
    """
    return Settings(
        vectorstore={"provider": "in_memory", "collection_name": "test_collection"},
        embedding={"provider": "sentence_transformer", "model": "all-MiniLM-L6-v2", "batch_size": 4},
        chunker={"strategy": "ast", "chunk_size": 256, "chunk_overlap": 32, "min_chunk_size": 5},
    )


# ---------------------------------------------------------------------------
# Source files
# ---------------------------------------------------------------------------


PYTHON_SAMPLE = '''
def add(a: int, b: int) -> int:
    """Return the sum of a and b."""
    return a + b


class Calculator:
    """A simple calculator."""

    def __init__(self) -> None:
        self.history: list[int] = []

    def multiply(self, x: int, y: int) -> int:
        result = x * y
        self.history.append(result)
        return result
'''

JAVASCRIPT_SAMPLE = '''
function greet(name) {
  return `Hello, ${name}!`;
}

class Greeter {
  constructor(prefix) {
    this.prefix = prefix;
  }

  greet(name) {
    return `${this.prefix}, ${name}!`;
  }
}

module.exports = { greet, Greeter };
'''


@pytest.fixture
def python_source_file() -> SourceFile:
    """A minimal Python source file for testing."""
    return SourceFile(
        path="sample/math.py",
        content=PYTHON_SAMPLE.strip(),
        language=Language.PYTHON,
    )


@pytest.fixture
def javascript_source_file() -> SourceFile:
    """A minimal JavaScript source file for testing."""
    return SourceFile(
        path="sample/greeter.js",
        content=JAVASCRIPT_SAMPLE.strip(),
        language=Language.JAVASCRIPT,
    )


# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------


@pytest.fixture
def in_memory_store() -> InMemoryVectorStore:
    """A fresh in-memory vector store for each test."""
    return InMemoryVectorStore(collection_name="test", dimensions=4)
