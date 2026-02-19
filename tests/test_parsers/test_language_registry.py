"""Tests for language detection."""

from __future__ import annotations

import pytest

from code_indexer.core.models import Language
from code_indexer.parsers.language_registry import (
    detect_language,
    register_extension,
    supported_extensions,
)


@pytest.mark.parametrize(
    "path,expected",
    [
        ("main.py", Language.PYTHON),
        ("app.js", Language.JAVASCRIPT),
        ("component.tsx", Language.TYPESCRIPT),
        ("lib.rs", Language.RUST),
        ("server.go", Language.GO),
        ("Main.java", Language.JAVA),
        ("utils.cpp", Language.CPP),
        ("header.h", Language.C),
        ("README.md", Language.UNKNOWN),
        ("data.json", Language.UNKNOWN),
        ("no_extension", Language.UNKNOWN),
    ],
)
def test_detect_language(path: str, expected: Language) -> None:
    assert detect_language(path) == expected


def test_register_custom_extension() -> None:
    """Custom extensions can be registered at runtime."""
    register_extension(".svelte", Language.JAVASCRIPT)
    assert detect_language("App.svelte") == Language.JAVASCRIPT


def test_supported_extensions_is_sorted() -> None:
    exts = supported_extensions()
    assert exts == sorted(exts)
