"""
Language registry: maps file extensions → Language enum values and loads
Tree-sitter grammar objects on demand.

Design notes
------------
* Grammars are loaded lazily and cached in a module-level dict so each
  grammar package is imported at most once per process.
* ``detect_language`` is intentionally non-raising: it returns
  ``Language.UNKNOWN`` for files it cannot identify rather than throwing,
  letting the pipeline decide whether to skip or attempt generic chunking.
* The extension map can be extended at runtime by calling
  ``register_extension``.
"""

from __future__ import annotations

import importlib
import logging
from pathlib import Path

from code_indexer.core.models import Language

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Extension → Language map
# ---------------------------------------------------------------------------
# We deliberately list common variations (e.g. .ts and .tsx both → TYPESCRIPT)
# and keep the dict ordered so more specific entries can be placed first.

_EXT_MAP: dict[str, Language] = {
    # Python
    ".py": Language.PYTHON,
    ".pyw": Language.PYTHON,
    ".pyi": Language.PYTHON,
    # JavaScript
    ".js": Language.JAVASCRIPT,
    ".jsx": Language.JAVASCRIPT,
    ".mjs": Language.JAVASCRIPT,
    ".cjs": Language.JAVASCRIPT,
    # TypeScript
    ".ts": Language.TYPESCRIPT,
    ".tsx": Language.TYPESCRIPT,
    ".mts": Language.TYPESCRIPT,
    ".cts": Language.TYPESCRIPT,
    # Rust
    ".rs": Language.RUST,
    # Go
    ".go": Language.GO,
    # Java
    ".java": Language.JAVA,
    # C++
    ".cpp": Language.CPP,
    ".cxx": Language.CPP,
    ".cc": Language.CPP,
    ".hpp": Language.CPP,
    ".hxx": Language.CPP,
    ".hh": Language.CPP,
    # C
    ".c": Language.C,
    ".h": Language.C,
}

# Grammar module names for each supported language.
# These correspond to the PyPI packages installed via pyproject.toml.
_GRAMMAR_MODULES: dict[Language, str] = {
    Language.PYTHON: "tree_sitter_python",
    Language.JAVASCRIPT: "tree_sitter_javascript",
    Language.TYPESCRIPT: "tree_sitter_typescript",
    Language.RUST: "tree_sitter_rust",
    Language.GO: "tree_sitter_go",
    Language.JAVA: "tree_sitter_java",
    Language.CPP: "tree_sitter_cpp",
    Language.C: "tree_sitter_c",
}

# Cache: Language → tree_sitter.Language object (loaded grammar)
_grammar_cache: dict[Language, object] = {}


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def detect_language(path: str | Path) -> Language:
    """Detect the programming language of a file from its extension.

    Args:
        path: File path (only the extension is examined).

    Returns:
        A ``Language`` enum value, or ``Language.UNKNOWN`` if unrecognised.

    Examples::

        >>> detect_language("src/auth.py")
        <Language.PYTHON: 'python'>
        >>> detect_language("README.md")
        <Language.UNKNOWN: 'unknown'>
    """
    ext = Path(path).suffix.lower()
    return _EXT_MAP.get(ext, Language.UNKNOWN)


def get_tree_sitter_language(language: Language) -> object | None:
    """Return the cached Tree-sitter ``Language`` object for *language*.

    Tree-sitter ``Language`` objects are built from grammar shared libraries
    (.so/.dylib/.dll) bundled inside the grammar PyPI packages.  We import
    the package and call its ``language()`` factory function.

    Args:
        language: The target ``Language`` enum value.

    Returns:
        A ``tree_sitter.Language`` instance, or ``None`` if the grammar
        package is not installed or the language is ``UNKNOWN``.

    .. note::
        This function is intentionally non-raising so that the parser can
        fall back gracefully when an optional grammar package is missing.
    """
    if language is Language.UNKNOWN:
        return None

    if language in _grammar_cache:
        return _grammar_cache[language]

    module_name = _GRAMMAR_MODULES.get(language)
    if not module_name:
        logger.debug("No grammar module registered for language %s", language.value)
        return None

    try:
        import tree_sitter  # noqa: PLC0415

        module = importlib.import_module(module_name)

        # TypeScript is special: the package exposes two grammars –
        # ``language_typescript`` and ``language_tsx``.
        if language is Language.TYPESCRIPT:
            ts_lang = tree_sitter.Language(module.language_typescript())
        else:
            ts_lang = tree_sitter.Language(module.language())

        _grammar_cache[language] = ts_lang
        logger.debug("Loaded Tree-sitter grammar for %s", language.value)
        return ts_lang

    except ImportError:
        logger.warning(
            "Grammar package %r not installed; language %s will be skipped",
            module_name,
            language.value,
        )
        return None
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Failed to load Tree-sitter grammar for %s: %s",
            language.value,
            exc,
        )
        return None


def register_extension(ext: str, language: Language) -> None:
    """Register a custom file extension at runtime.

    Args:
        ext:      File extension including the leading dot, e.g. ``".ts"``.
        language: The ``Language`` to associate with this extension.

    Example::

        register_extension(".svelte", Language.JAVASCRIPT)
    """
    if not ext.startswith("."):
        ext = "." + ext
    _EXT_MAP[ext.lower()] = language
    logger.debug("Registered extension %r → %s", ext, language.value)


def supported_extensions() -> list[str]:
    """Return a sorted list of all registered file extensions."""
    return sorted(_EXT_MAP.keys())


def supported_languages() -> list[Language]:
    """Return the list of languages that have Tree-sitter grammars registered."""
    return list(_GRAMMAR_MODULES.keys())
