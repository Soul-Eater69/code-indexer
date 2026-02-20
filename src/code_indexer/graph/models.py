"""Pydantic models for the code knowledge graph.

Node types
----------
* :class:`FileNode`      — a source file in the repository
* :class:`SymbolNode`    — a named code symbol (function / class / method / …)
* :class:`DirectoryNode` — a directory in the repository tree

Edge / relationship types
-------------------------
* ``IMPORTS``        — ``(:File)→[:IMPORTS]→(:File)``
* ``DEFINES``        — ``(:File)→[:DEFINES]→(:Symbol)``
* ``CONTAINS``       — ``(:Symbol)→[:CONTAINS]→(:Symbol)``  (class→method)
* ``CALLS``          — ``(:Symbol)→[:CALLS]→(:Symbol)``
* ``INHERITS_FROM``  — ``(:Symbol)→[:INHERITS_FROM]→(:Symbol)``
* ``REFERENCES``     — ``(:Symbol)→[:REFERENCES]→(:Symbol)``
* ``PART_OF``        — ``(:File/:Directory)→[:PART_OF]→(:Directory)``
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator

from code_indexer.core.models import Language


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class NodeType(str, Enum):
    """Types of nodes in the code knowledge graph."""

    FILE = "File"
    SYMBOL = "Symbol"
    DIRECTORY = "Directory"


class SymbolKind(str, Enum):
    """Fine-grained classification of a :class:`SymbolNode`."""

    FUNCTION = "function"
    CLASS = "class"
    METHOD = "method"
    INTERFACE = "interface"
    ENUM = "enum"
    VARIABLE = "variable"
    CONSTANT = "constant"
    DECORATOR = "decorator"
    STRUCT = "struct"
    TRAIT = "trait"
    MODULE = "module"
    UNKNOWN = "unknown"


class RelType(str, Enum):
    """Types of directed edges in the code knowledge graph."""

    IMPORTS = "IMPORTS"
    DEFINES = "DEFINES"
    CONTAINS = "CONTAINS"
    CALLS = "CALLS"
    INHERITS_FROM = "INHERITS_FROM"
    REFERENCES = "REFERENCES"
    PART_OF = "PART_OF"


# ---------------------------------------------------------------------------
# Node models
# ---------------------------------------------------------------------------


class FileNode(BaseModel):
    """A source file in the repository.

    Attributes:
        id:           Deterministic SHA-256 of ``path`` (repo-relative).
        path:         Repo-relative POSIX path (``src/auth/jwt.py``).
        language:     Detected programming language.
        sha256:       SHA-256 of file *content* (for change detection).
        node_type:    Always ``NodeType.FILE``.
    """

    id: str
    path: str
    language: Language
    sha256: str
    node_type: NodeType = NodeType.FILE

    @classmethod
    def from_path(cls, path: str, language: Language, sha256: str) -> "FileNode":
        node_id = hashlib.sha256(path.encode()).hexdigest()[:16]
        return cls(id=node_id, path=path, language=language, sha256=sha256)


class SymbolNode(BaseModel):
    """A named code symbol (function, class, method, …).

    Attributes:
        id:             Deterministic ID from ``file_id + ":" + qualified_name``.
        name:           Short name, e.g. ``"verify_token"``.
        qualified_name: Full dotted name, e.g. ``"auth.jwt.JWTHandler.verify_token"``.
                        For top-level symbols this is ``"module.name"``.
        kind:           Fine-grained symbol kind.
        file_id:        ID of the :class:`FileNode` that defines this symbol.
        file_path:      Convenience copy of the file path (avoids joins).
        start_line:     1-indexed start line in the source file.
        end_line:       1-indexed end line in the source file.
        parent_name:    For methods/nested functions: short name of the
                        containing class or function.  ``None`` for top-level.
        docstring:      First docstring / leading comment, if extracted.
        node_type:      Always ``NodeType.SYMBOL``.
    """

    id: str
    name: str
    qualified_name: str
    kind: SymbolKind
    file_id: str
    file_path: str
    start_line: int
    end_line: int
    parent_name: str | None = None
    docstring: str | None = None
    node_type: NodeType = NodeType.SYMBOL

    @classmethod
    def make_id(cls, file_id: str, qualified_name: str) -> str:
        raw = f"{file_id}:{qualified_name}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


class DirectoryNode(BaseModel):
    """A directory in the repository file tree.

    Attributes:
        id:   Deterministic SHA-256 of ``path``.
        path: Repo-relative POSIX path (``src/auth``).
        name: Base name of the directory (``"auth"``).
        node_type: Always ``NodeType.DIRECTORY``.
    """

    id: str
    path: str
    name: str
    node_type: NodeType = NodeType.DIRECTORY

    @classmethod
    def from_path(cls, path: str) -> "DirectoryNode":
        node_id = hashlib.sha256(path.encode()).hexdigest()[:16]
        name = path.rstrip("/").rsplit("/", 1)[-1] or path
        return cls(id=node_id, path=path, name=name)


# ---------------------------------------------------------------------------
# Edge models
# ---------------------------------------------------------------------------


class GraphEdge(BaseModel):
    """A directed relationship between two nodes.

    Attributes:
        id:         Deterministic ID from ``rel_type + source_id + target_id``.
        rel_type:   The relationship type.
        source_id:  ID of the source node.
        target_id:  ID of the target node.
        properties: Arbitrary extra data (line numbers, counts, …).
    """

    id: str
    rel_type: RelType
    source_id: str
    target_id: str
    properties: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def make_id(cls, rel_type: RelType, source_id: str, target_id: str) -> str:
        raw = f"{rel_type.value}:{source_id}:{target_id}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    @classmethod
    def create(
        cls,
        rel_type: RelType,
        source_id: str,
        target_id: str,
        **props: Any,
    ) -> "GraphEdge":
        return cls(
            id=cls.make_id(rel_type, source_id, target_id),
            rel_type=rel_type,
            source_id=source_id,
            target_id=target_id,
            properties=props,
        )


# ---------------------------------------------------------------------------
# Composite / result models
# ---------------------------------------------------------------------------


class GraphSnapshot(BaseModel):
    """Complete extracted graph for a repository.

    Produced by :class:`~code_indexer.graph.pipeline.GraphIndexingPipeline`
    and consumed by graph store ``upsert`` methods.

    Attributes:
        file_nodes:      ``{node_id: FileNode}``
        symbol_nodes:    ``{node_id: SymbolNode}``
        directory_nodes: ``{node_id: DirectoryNode}``
        edges:           All directed edges (any type).
        indexed_at:      UTC timestamp of when this snapshot was built.
    """

    file_nodes: dict[str, FileNode] = Field(default_factory=dict)
    symbol_nodes: dict[str, SymbolNode] = Field(default_factory=dict)
    directory_nodes: dict[str, DirectoryNode] = Field(default_factory=dict)
    edges: list[GraphEdge] = Field(default_factory=list)
    indexed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Convenience counters

    @property
    def num_files(self) -> int:
        return len(self.file_nodes)

    @property
    def num_symbols(self) -> int:
        return len(self.symbol_nodes)

    @property
    def num_edges(self) -> int:
        return len(self.edges)

    def edges_of_type(self, rel_type: RelType) -> list[GraphEdge]:
        return [e for e in self.edges if e.rel_type == rel_type]


class SymbolContext(BaseModel):
    """Full structural context for a single symbol.

    Returned by the graph store's ``get_symbol_context`` query.  Contains
    everything needed for RAG-style prompt construction:

    - The symbol itself and the file it lives in.
    - Caller/callee relationships (call graph).
    - Inheritance chain (class hierarchy).
    - Methods defined inside (for classes).
    - Files that import the symbol's file.
    """

    symbol: SymbolNode
    file: FileNode
    callers: list[SymbolNode] = Field(default_factory=list)
    callees: list[SymbolNode] = Field(default_factory=list)
    parent_class: SymbolNode | None = None
    methods: list[SymbolNode] = Field(default_factory=list)
    inherits_from: list[SymbolNode] = Field(default_factory=list)
    subclasses: list[SymbolNode] = Field(default_factory=list)
    file_imports: list[FileNode] = Field(default_factory=list)
    file_imported_by: list[FileNode] = Field(default_factory=list)

    def to_context_text(self) -> str:
        """Render a human-readable structural context string for LLM prompts."""
        lines: list[str] = [
            f"Symbol: {self.symbol.qualified_name} [{self.symbol.kind.value}]",
            f"File: {self.file.path} (lines {self.symbol.start_line}–{self.symbol.end_line})",
        ]
        if self.parent_class:
            lines.append(f"Member of class: {self.parent_class.qualified_name}")
        if self.inherits_from:
            names = ", ".join(s.name for s in self.inherits_from)
            lines.append(f"Inherits from: {names}")
        if self.subclasses:
            names = ", ".join(s.name for s in self.subclasses)
            lines.append(f"Subclasses: {names}")
        if self.callees:
            names = ", ".join(s.name for s in self.callees[:10])
            lines.append(f"Calls: {names}")
        if self.callers:
            names = ", ".join(s.name for s in self.callers[:10])
            lines.append(f"Called by: {names}")
        if self.methods:
            names = ", ".join(s.name for s in self.methods[:10])
            lines.append(f"Methods: {names}")
        if self.file_imports:
            paths = ", ".join(f.path for f in self.file_imports[:5])
            lines.append(f"File imports: {paths}")
        return "\n".join(lines)


class GraphStats(BaseModel):
    """Statistics about the current graph index."""

    total_files: int = 0
    total_symbols: int = 0
    total_directories: int = 0
    total_edges: int = 0
    edges_by_type: dict[str, int] = Field(default_factory=dict)
    symbols_by_kind: dict[str, int] = Field(default_factory=dict)
    languages: dict[str, int] = Field(default_factory=dict)
    graph_store: str = "unknown"
    indexed_at: datetime | None = None


# ---------------------------------------------------------------------------
# Intermediate extraction results (internal, not stored in graph)
# ---------------------------------------------------------------------------


class RawImport(BaseModel):
    """Raw import statement extracted from source before resolution."""

    module_string: str
    """Original import string: ``"os.path"``, ``".utils"``, ``"react"``."""

    imported_names: list[str] = Field(default_factory=list)
    """Specific names imported: ``["join", "exists"]``, or ``["*"]``, or ``[]``
    for bare module imports."""

    alias: str | None = None
    """Alias: ``"np"`` in ``import numpy as np``."""

    is_relative: bool = False
    """True for Python ``from . import x`` or JS ``from './x'`` imports."""

    line: int = 0


class RawCall(BaseModel):
    """Raw call expression extracted from source before resolution."""

    callee_name: str
    """Short function/method name being called."""

    callee_object: str | None = None
    """Object/module prefix: ``"obj"`` in ``obj.method()``."""

    caller_qualified_name: str = ""
    """Qualified name of the enclosing function (best-effort, may be empty)."""

    line: int = 0


class RawInheritance(BaseModel):
    """Raw class inheritance extracted from source before resolution."""

    class_name: str
    base_names: list[str]
    line: int = 0


class RawSymbol(BaseModel):
    """Raw symbol definition extracted before full qualification."""

    name: str
    kind: SymbolKind
    start_line: int
    end_line: int
    parent_name: str | None = None
    """Containing class name for methods, ``None`` for top-level."""


class FileExtractionResult(BaseModel):
    """All raw graph data extracted from a single source file."""

    file_path: str
    language: Language
    symbols: list[RawSymbol] = Field(default_factory=list)
    imports: list[RawImport] = Field(default_factory=list)
    calls: list[RawCall] = Field(default_factory=list)
    inheritance: list[RawInheritance] = Field(default_factory=list)
