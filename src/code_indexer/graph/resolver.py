"""Import path resolver.

Maps raw import strings (``"from .utils import foo"`` / ``"./utils"`` /
``"github.com/user/pkg/auth"``) to concrete file nodes in the indexed
codebase.

Resolution strategy
-------------------
1. **Relative imports** (``./``, ``../``, Python ``from . import …``):
   Resolved by joining the importing file's directory with the import path,
   then trying candidate extensions in priority order.

2. **Absolute imports within the repo** (``mypackage.auth``, ``auth/utils``):
   Resolved by converting the module path to filesystem components and
   searching the known file map.  Python uses ``.`` separators; others use
   ``/`` or ``::`` separators.

3. **External packages** (``react``, ``numpy``, ``std::collections``):
   Unresolvable within the repo.  ``resolve_import`` returns ``None``
   and the caller should set ``is_external=True``.

All lookups are O(1) dictionary lookups against a ``{path: FileNode}`` map
that is built once per indexing run.
"""

from __future__ import annotations

import posixpath
from typing import TYPE_CHECKING

from code_indexer.core.models import Language

if TYPE_CHECKING:
    from code_indexer.graph.models import FileNode

# ---------------------------------------------------------------------------
# Extension candidates per language
# (ordered by priority when resolving bare import paths)
# ---------------------------------------------------------------------------

_PY_EXTENSIONS = [".py", "/__init__.py", ".pyi"]
_JS_EXTENSIONS = [".js", ".jsx", ".ts", ".tsx", "/index.js", "/index.ts", "/index.jsx", "/index.tsx"]
_GO_EXTENSIONS = []  # Go imports are module paths, not file paths
_RUST_EXTENSIONS = [".rs", "/mod.rs"]
_JAVA_EXTENSIONS = [".java"]
_CPP_EXTENSIONS = [".h", ".hpp", ".hxx", ".hh", ".cpp", ".cc", ".cxx"]

_LANG_EXTENSIONS: dict[Language, list[str]] = {
    Language.PYTHON: _PY_EXTENSIONS,
    Language.JAVASCRIPT: _JS_EXTENSIONS,
    Language.TYPESCRIPT: _JS_EXTENSIONS,
    Language.RUST: _RUST_EXTENSIONS,
    Language.JAVA: _JAVA_EXTENSIONS,
    Language.CPP: _CPP_EXTENSIONS,
    Language.C: _CPP_EXTENSIONS,
}


class ImportResolver:
    """Resolve import strings to :class:`~code_indexer.graph.models.FileNode` IDs.

    Parameters
    ----------
    file_map:
        ``{repo_relative_path: FileNode}`` — all files in the current index.
        Built by :class:`~code_indexer.graph.pipeline.GraphIndexingPipeline`.
    """

    def __init__(self, file_map: dict[str, "FileNode"]) -> None:
        # Normalise all keys to POSIX paths without leading ./
        self._file_map = {
            posixpath.normpath(k).lstrip("./"): v for k, v in file_map.items()
        }
        # Stem index: strip extension → list[path]  (used by _by_stem)
        self._by_stem: dict[str, list[str]] = {}
        # Suffix index (GitNexus pattern): every trailing sub-path of a stem
        # maps to the file.  Allows O(1) resolution of partial import paths
        # like ``from auth import jwt`` finding ``myapp/auth/jwt.py``.
        # Built once at construction time; lookups are O(1) dict accesses.
        self._suffix_index: dict[str, list["FileNode"]] = {}
        for path in self._file_map:
            stem = posixpath.splitext(path)[0]
            self._by_stem.setdefault(stem, []).append(path)
            # Register every suffix of the stem
            parts = stem.split("/")
            for i in range(len(parts)):
                suffix = "/".join(parts[i:])
                self._suffix_index.setdefault(suffix, []).append(self._file_map[path])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resolve(
        self,
        module_string: str,
        importing_file_path: str,
        language: Language,
    ) -> "FileNode | None":
        """Resolve ``module_string`` to a :class:`FileNode`, or ``None``.

        Parameters
        ----------
        module_string:
            The raw import string as found in source code.
        importing_file_path:
            Repo-relative path of the file that contains the import.
        language:
            Language of the importing file (drives resolution heuristics).
        """
        if language == Language.PYTHON:
            return self._resolve_python(module_string, importing_file_path)
        elif language in (Language.JAVASCRIPT, Language.TYPESCRIPT):
            return self._resolve_js(module_string, importing_file_path)
        elif language == Language.RUST:
            return self._resolve_rust(module_string, importing_file_path)
        elif language == Language.JAVA:
            return self._resolve_java(module_string)
        else:
            return None

    # ------------------------------------------------------------------
    # Language-specific resolvers
    # ------------------------------------------------------------------

    def _resolve_python(
        self, module_string: str, importing_path: str
    ) -> "FileNode | None":
        """Resolve a Python import string.

        Examples::

            "os.path"           → external
            ".utils"            → relative: <dir>/utils.py
            "..models"          → relative: <parent>/models.py
            "mypackage.auth"    → absolute: mypackage/auth.py
        """
        if module_string.startswith("."):
            # Relative import — count leading dots
            dots = len(module_string) - len(module_string.lstrip("."))
            rest = module_string[dots:]

            current_dir = posixpath.dirname(importing_path)
            for _ in range(dots - 1):
                current_dir = posixpath.dirname(current_dir)

            # Convert dotted name to path
            if rest:
                rel_path = posixpath.join(current_dir, rest.replace(".", "/"))
            else:
                rel_path = current_dir

            return self._try_extensions(rel_path, Language.PYTHON)
        else:
            # Absolute import — try converting dots to path separators
            as_path = module_string.replace(".", "/")
            result = self._try_extensions(as_path, Language.PYTHON)
            if result:
                return result
            # Also try finding the package root
            parts = module_string.split(".")
            for i in range(len(parts), 0, -1):
                candidate = "/".join(parts[:i])
                result = self._try_extensions(candidate, Language.PYTHON)
                if result:
                    return result
            # Final fallback: suffix index (O(1), handles deep monorepo nesting)
            return self._try_suffix_index(module_string)

    def _resolve_js(
        self, module_string: str, importing_path: str
    ) -> "FileNode | None":
        """Resolve a JS/TS import string.

        Examples::

            "./utils"           → relative: <dir>/utils.{js,ts,…}
            "../auth/index"     → relative: <parent>/auth/index.{js,ts,…}
            "react"             → external → None
        """
        if not module_string.startswith("."):
            # External package
            return None

        current_dir = posixpath.dirname(importing_path)
        joined = posixpath.normpath(posixpath.join(current_dir, module_string))
        return self._try_extensions(joined, Language.JAVASCRIPT)

    def _resolve_rust(
        self, module_string: str, importing_path: str
    ) -> "FileNode | None":
        """Resolve a Rust ``use`` path.

        Examples::

            "crate::auth::jwt"      → src/auth/jwt.rs
            "self::utils"           → <dir>/utils.rs
            "super::models"         → <parent>/models.rs
            "std::collections::HashMap" → external
        """
        if module_string.startswith("std::") or module_string.startswith("core::"):
            return None  # Standard library

        current_dir = posixpath.dirname(importing_path)

        if module_string.startswith("self::"):
            parts = module_string[len("self::") :].split("::")
            path = posixpath.join(current_dir, *parts)
            return self._try_extensions(path, Language.RUST)

        if module_string.startswith("super::"):
            parts = module_string[len("super::") :].split("::")
            parent_dir = posixpath.dirname(current_dir)
            path = posixpath.join(parent_dir, *parts)
            return self._try_extensions(path, Language.RUST)

        if module_string.startswith("crate::"):
            # Assume crate root is src/
            parts = module_string[len("crate::") :].split("::")
            path = posixpath.join("src", *parts)
            return self._try_extensions(path, Language.RUST)

        # Try as a direct path from repo root
        parts = module_string.split("::")
        path = posixpath.join(*parts)
        return self._try_extensions(path, Language.RUST)

    def _resolve_java(self, module_string: str) -> "FileNode | None":
        """Resolve a Java import declaration.

        ``com.example.myapp.auth.JWTHandler`` → ``com/example/myapp/auth/JWTHandler.java``
        """
        # Strip wildcard
        clean = module_string.rstrip(".*")
        as_path = clean.replace(".", "/")
        result = self._try_extensions(as_path, Language.JAVA)
        if result:
            return result
        # Fallback: suffix index handles partial package paths
        return self._try_suffix_index(clean)

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    def _try_extensions(
        self, base_path: str, language: Language
    ) -> "FileNode | None":
        """Try ``base_path`` + each candidate extension until a match is found."""
        norm = posixpath.normpath(base_path)

        # Exact match first
        if norm in self._file_map:
            return self._file_map[norm]

        for ext in _LANG_EXTENSIONS.get(language, []):
            candidate = posixpath.normpath(norm + ext)
            if candidate in self._file_map:
                return self._file_map[candidate]

        return None

    def _try_suffix_index(self, module_string: str) -> "FileNode | None":
        """Look up ``module_string`` in the suffix index.

        Converts dotted or slash-separated import paths to a suffix key
        and returns the unique matching :class:`FileNode`, or ``None`` if
        no match or multiple matches exist (ambiguous).

        This enables O(1) resolution of imports like ``import auth.jwt``
        when the file lives at ``myapp/services/auth/jwt.py`` without
        iterating the entire file map.
        """
        # Normalise: dots → slashes (Python / Java style)
        key = module_string.replace(".", "/").strip("/")
        candidates = self._suffix_index.get(key)
        if candidates and len(candidates) == 1:
            return candidates[0]
        return None
