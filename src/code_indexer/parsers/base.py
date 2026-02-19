"""
Abstract base class for all parsers.

Every parser, regardless of the underlying technology (Tree-sitter, regex,
heuristic), must implement ``parse``.  The rest of the pipeline works
exclusively through this interface, making it trivial to swap or add parsers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from code_indexer.core.models import ParsedFile, SourceFile


class BaseParser(ABC):
    """Abstract parser interface.

    Subclasses only need to implement :meth:`parse`.  The base class
    provides no shared state so instances are thread-safe and reusable.
    """

    @abstractmethod
    def parse(self, source_file: SourceFile) -> ParsedFile:
        """Parse *source_file* and return a ``ParsedFile``.

        Implementations MUST NOT raise on parse errors that Tree-sitter
        can recover from (it is error-tolerant by design).  Only raise for
        unrecoverable errors like I/O failures or OOM conditions.

        Args:
            source_file: The file to parse.

        Returns:
            A ``ParsedFile`` containing extracted AST nodes and any
            non-fatal error messages encountered during parsing.
        """

    @property
    def name(self) -> str:
        """Human-readable name of this parser implementation."""
        return type(self).__name__
