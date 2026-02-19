"""
Codebase file walker.

The walker's job is to traverse a directory tree and yield ``SourceFile``
objects for every source file it finds.  It handles:
  * Glob-pattern based ignore rules (like ``.gitignore`` semantics).
  * File size limits.
  * Character encoding detection (chardet).
  * Language detection from file extensions.
  * Skipping binary files.

Design: we use ``pathlib.Path.rglob("*")`` instead of ``os.walk`` because it
is easier to compose with glob patterns and produces consistent cross-platform
behaviour.
"""

from __future__ import annotations

import fnmatch
import logging
from pathlib import Path
from typing import Generator, Iterator

from code_indexer.core.models import Language, SourceFile
from code_indexer.parsers.language_registry import detect_language

logger = logging.getLogger(__name__)


def _is_binary(content_bytes: bytes) -> bool:
    """Heuristic to detect binary files by looking for null bytes.

    This is the same approach used by Git.  We check only the first 8000
    bytes to keep it fast for large files.

    Args:
        content_bytes: Raw file bytes.

    Returns:
        ``True`` if the file appears to be binary.
    """
    sample = content_bytes[:8000]
    return b"\x00" in sample


def _detect_encoding(content_bytes: bytes) -> str:
    """Detect the character encoding of *content_bytes* using chardet.

    Falls back to ``utf-8`` if chardet is unavailable or uncertain.

    Args:
        content_bytes: Raw bytes to inspect.

    Returns:
        Encoding name string (e.g. ``"utf-8"``, ``"latin-1"``).
    """
    try:
        import chardet  # noqa: PLC0415

        result = chardet.detect(content_bytes)
        detected = result.get("encoding") or "utf-8"
        # Normalise aliases (chardet sometimes returns "ascii" for UTF-8 files).
        if detected.lower() in ("ascii", "us-ascii"):
            detected = "utf-8"
        return detected
    except ImportError:
        return "utf-8"


class CodebaseWalker:
    """Walk a directory tree and yield ``SourceFile`` objects.

    Args:
        root:             Absolute path to the codebase root directory.
        ignore_patterns:  List of glob patterns (relative to *root*) to skip.
                          Use ``"**/"``-prefixed patterns for recursive matching.
        max_file_size:    Skip files larger than this many bytes.
        include_unknown:  If ``True``, yield files with ``Language.UNKNOWN``
                          (e.g. plain-text files).  Default is ``False``.
        extra_metadata:   Key-value pairs to attach to every ``SourceFile``.
    """

    def __init__(
        self,
        root: str | Path,
        ignore_patterns: list[str] | None = None,
        max_file_size: int = 1_048_576,
        include_unknown: bool = False,
        extra_metadata: dict | None = None,
    ) -> None:
        self.root = Path(root).resolve()
        self.ignore_patterns = ignore_patterns or []
        self.max_file_size = max_file_size
        self.include_unknown = include_unknown
        self.extra_metadata = extra_metadata or {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def walk(self) -> Iterator[SourceFile]:
        """Yield ``SourceFile`` objects for every indexed source file.

        Files are yielded in filesystem order (deterministic on most OSes
        for reproducible indexing runs).

        Yields:
            ``SourceFile`` objects ready to be passed to a parser.
        """
        if not self.root.is_dir():
            raise FileNotFoundError(f"Root directory not found: {self.root}")

        for path in sorted(self.root.rglob("*")):
            if not path.is_file():
                continue

            rel_path = path.relative_to(self.root)

            if self._should_ignore(rel_path):
                logger.debug("Ignoring %s", rel_path)
                continue

            file_size = path.stat().st_size
            if file_size > self.max_file_size:
                logger.debug(
                    "Skipping %s: %d bytes > limit %d",
                    rel_path,
                    file_size,
                    self.max_file_size,
                )
                continue

            language = detect_language(path)
            if language is Language.UNKNOWN and not self.include_unknown:
                continue

            source_file = self._read_file(path, rel_path, language, file_size)
            if source_file is not None:
                yield source_file

    def count_files(self) -> int:
        """Count the number of files that would be walked (without reading content).

        Useful for progress bar initialisation.

        Returns:
            Number of files that pass all filters.
        """
        total = 0
        for path in self.root.rglob("*"):
            if not path.is_file():
                continue
            rel_path = path.relative_to(self.root)
            if self._should_ignore(rel_path):
                continue
            if path.stat().st_size > self.max_file_size:
                continue
            if detect_language(path) is Language.UNKNOWN and not self.include_unknown:
                continue
            total += 1
        return total

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _should_ignore(self, rel_path: Path) -> bool:
        """Return ``True`` if *rel_path* matches any ignore pattern.

        We check both the full relative path string and each individual path
        component so that patterns like ``node_modules/**`` work as expected.

        Args:
            rel_path: Path relative to the codebase root.

        Returns:
            ``True`` if the file should be skipped.
        """
        rel_str = str(rel_path).replace("\\", "/")
        parts = list(rel_path.parts)

        for pattern in self.ignore_patterns:
            # Check full path.
            if fnmatch.fnmatch(rel_str, pattern):
                return True
            # Check individual components (e.g. "node_modules/**" should
            # match any path that has "node_modules" as a component).
            for part in parts:
                if fnmatch.fnmatch(part, pattern.rstrip("/**").rstrip("/*")):
                    return True

        return False

    def _read_file(
        self,
        path: Path,
        rel_path: Path,
        language: Language,
        file_size: int,
    ) -> SourceFile | None:
        """Read a file and return a ``SourceFile``, or ``None`` on failure.

        Args:
            path:      Absolute file path.
            rel_path:  Path relative to codebase root.
            language:  Detected language.
            file_size: File size in bytes.

        Returns:
            ``SourceFile`` or ``None`` if the file is binary or unreadable.
        """
        try:
            raw_bytes = path.read_bytes()
        except OSError as exc:
            logger.warning("Cannot read %s: %s", path, exc)
            return None

        if _is_binary(raw_bytes):
            logger.debug("Skipping binary file: %s", rel_path)
            return None

        encoding = _detect_encoding(raw_bytes)
        try:
            content = raw_bytes.decode(encoding, errors="replace")
        except LookupError:
            content = raw_bytes.decode("utf-8", errors="replace")
            encoding = "utf-8"

        if not content.strip():
            logger.debug("Skipping empty file: %s", rel_path)
            return None

        return SourceFile(
            path=str(rel_path),
            content=content,
            language=language,
            encoding=encoding,
            size_bytes=file_size,
            repo_root=str(self.root),
            metadata=dict(self.extra_metadata),
        )
