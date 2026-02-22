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
* ``INJECTS``        — ``(:Symbol)→[:INJECTS]→(:Symbol)``  (class→dependency type)
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
    INJECTS = "INJECTS"
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
        decorators:     Decorator / annotation names applied to this symbol,
                        e.g. ``["staticmethod", "router.get", "pytest.mark.skip"]``.
                        Empty list when no decorators are present.
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
    decorators: list[str] = Field(default_factory=list)
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

    def compute_katz_centrality(
        self,
        alpha: float = 0.1,
        beta: float = 1.0,
        max_iter: int = 100,
        tol: float = 1e-6,
        edge_types: frozenset | None = None,
    ) -> dict[str, float]:
        """Compute Katz centrality for all symbol nodes in the snapshot.

        Katz centrality assigns higher scores to symbols that are
        transitively depended upon by many others — the architectural
        **hotspots** of the codebase.  It is the graph-theoretic analogue
        of ``entity_rank`` from the ``code-chopper`` library.

        The score for node ``v`` is:

        .. math::

            k_v = \\alpha \\sum_u A_{uv} \\, k_u + \\beta

        where :math:`A` is the adjacency matrix of the selected edge types
        (structural dependence direction: ``u → v`` means ``u`` depends on
        ``v``, so ``v`` gains centrality).  The iterative power-method
        converges in O(max_iter × |E|) time with no external dependencies.

        Parameters
        ----------
        alpha:
            Attenuation factor.  Must be less than ``1 / λ_max`` of the
            adjacency matrix.  The default of ``0.1`` is safe for
            typical code graphs.  Increase toward ``0.3`` for larger
            repos where you want multi-hop influence to propagate further.
        beta:
            Baseline score added at every iteration (default ``1.0``).
            Symbols with no incoming edges receive exactly ``beta`` in the
            first iteration.
        max_iter:
            Maximum power-method iterations (default ``100``).
        tol:
            Convergence tolerance: stop when the total change in scores
            across all nodes is less than ``tol × n`` (default ``1e-6``).
        edge_types:
            Set of :class:`RelType` values whose edges are included in the
            adjacency matrix.  Defaults to
            ``{CALLS, INJECTS, INHERITS_FROM}`` — the three structural
            dependency types.

        Returns
        -------
        dict[str, float]
            ``{symbol_id: score}`` normalised to ``[0, 1]``.  Symbols
            not in the snapshot return ``0.0`` when looked up.
        """
        if edge_types is None:
            edge_types = frozenset(
                {RelType.CALLS, RelType.INJECTS, RelType.INHERITS_FROM}
            )

        symbol_ids = list(self.symbol_nodes.keys())
        if not symbol_ids:
            return {}

        idx_of: dict[str, int] = {sid: i for i, sid in enumerate(symbol_ids)}
        n = len(symbol_ids)

        # in_adj[v] = list of source indices u such that u→v exists.
        # An edge u→v (u calls/injects/inherits v) means v gains centrality.
        in_adj: list[list[int]] = [[] for _ in range(n)]
        for edge in self.edges:
            if edge.rel_type not in edge_types:
                continue
            src_idx = idx_of.get(edge.source_id)
            tgt_idx = idx_of.get(edge.target_id)
            if src_idx is not None and tgt_idx is not None:
                in_adj[tgt_idx].append(src_idx)

        # Power iteration: k^(t+1)[v] = alpha * sum_u k^(t)[u] + beta
        k: list[float] = [beta] * n
        for _ in range(max_iter):
            k_new: list[float] = [beta] * n
            for v in range(n):
                for u in in_adj[v]:
                    k_new[v] += alpha * k[u]
            diff = sum(abs(k_new[i] - k[i]) for i in range(n))
            k = k_new
            if diff < tol * n:
                break

        # Normalise to [0, 1]
        lo, hi = min(k), max(k)
        span = hi - lo
        if span < 1e-10:
            return {sid: 1.0 for sid in symbol_ids}
        return {symbol_ids[i]: (k[i] - lo) / span for i in range(n)}


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
        if self.symbol.decorators:
            lines.append(f"Decorators: {', '.join(self.symbol.decorators)}")
        return "\n".join(lines)

    def to_mermaid(self) -> str:
        """Render a Mermaid flowchart of the call neighbourhood for LLM context.

        The diagram places the focal symbol in the centre, with callers above
        and callees below.  Inheritance and class membership are shown as
        separate edge styles.

        Example output::

            flowchart TD
                focal["verify_token [method]"]
                caller0["handle_request [function]"]
                caller0 -->|calls| focal
                callee0["decode_jwt [function]"]
                focal -->|calls| callee0
                parent["JWTHandler [class]"]
                parent -->|contains| focal
        """
        # Sanitise label text (no quotes inside Mermaid labels)
        def _lbl(sym: "SymbolNode") -> str:
            return f'{sym.name} [{sym.kind.value}]'.replace('"', "'")

        lines = ["flowchart TD"]
        focal_id = "focal"
        focal_label = _lbl(self.symbol)
        lines.append(f'    {focal_id}["{focal_label}"]')

        for i, caller in enumerate(self.callers[:8]):
            nid = f"caller{i}"
            lines.append(f'    {nid}["{_lbl(caller)}"]')
            lines.append(f"    {nid} -->|calls| {focal_id}")

        for i, callee in enumerate(self.callees[:8]):
            nid = f"callee{i}"
            lines.append(f'    {nid}["{_lbl(callee)}"]')
            lines.append(f"    {focal_id} -->|calls| {nid}")

        if self.parent_class:
            lines.append(f'    parent["{_lbl(self.parent_class)}"]')
            lines.append(f"    parent -->|contains| {focal_id}")

        for i, base in enumerate(self.inherits_from[:4]):
            nid = f"base{i}"
            lines.append(f'    {nid}["{_lbl(base)}"]')
            lines.append(f"    {focal_id} -->|inherits| {nid}")

        for i, sub in enumerate(self.subclasses[:4]):
            nid = f"sub{i}"
            lines.append(f'    {nid}["{_lbl(sub)}"]')
            lines.append(f"    {nid} -->|inherits| {focal_id}")

        return "\n".join(lines)


class ImpactResult(BaseModel):
    """Result of an impact analysis query.

    Answers: "If I change symbol X, what else could break?"

    Call-graph edges carry a confidence score (0.0–1.0) produced by the
    three-tier name-match resolver.  All lists respect the ``min_confidence``
    threshold that was applied when the query was issued.

    Attributes:
        symbol:              The symbol that is being changed.
        direct_callers:      Symbols with a direct ``CALLS`` edge to this symbol.
        transitive_callers:  All reachable callers (unlimited BFS, includes
                             direct callers).  Filtered by ``min_confidence``.
        subclasses:          Direct subclasses / implementors — relevant when
                             the changed symbol is a class or interface.
        importing_files:     Files that directly import the file containing
                             this symbol.  These always need review.
        affected_files:      Deduplicated set of files containing any symbol in
                             ``transitive_callers`` or ``subclasses``.  This is
                             the minimal set of files that should be retested.
        confidence_breakdown: Count of transitive callers by confidence tier:
                              ``certain`` (≥ 0.85), ``probable`` (0.50–0.84),
                              ``speculative`` (< 0.50).
        min_confidence_used: The threshold that was applied.
    """

    symbol: SymbolNode
    direct_callers: list[SymbolNode] = Field(default_factory=list)
    transitive_callers: list[SymbolNode] = Field(default_factory=list)
    subclasses: list[SymbolNode] = Field(default_factory=list)
    importing_files: list[FileNode] = Field(default_factory=list)
    affected_files: list[FileNode] = Field(default_factory=list)
    confidence_breakdown: dict[str, int] = Field(default_factory=dict)
    min_confidence_used: float = 0.0

    def to_mermaid(self) -> str:
        """Render a Mermaid flowchart of the full impact blast radius.

        The changed symbol is highlighted.  Direct callers appear one hop
        away; transitive-only callers are shown further out.  Subclasses are
        shown with a dashed inheritance arrow.

        Example output::

            flowchart TD
                target["◆ verify_token [method]  ← CHANGED"]:::changed
                dc0["handle_request [function]"]
                dc0 -->|calls| target
                tc0["middleware [function]"]
                tc0 -.->|transitive| target
                classDef changed fill:#f96,stroke:#c33,color:#000
        """
        def _lbl(sym: "SymbolNode") -> str:
            return f'{sym.name} [{sym.kind.value}]'.replace('"', "'")

        direct_ids: set[str] = {s.id for s in self.direct_callers}
        lines = ["flowchart TD"]
        lines.append(
            f'    target["◆ {_lbl(self.symbol)}  ← CHANGED"]:::changed'
        )

        for i, sym in enumerate(self.direct_callers[:12]):
            nid = f"dc{i}"
            lines.append(f'    {nid}["{_lbl(sym)}"]')
            lines.append(f"    {nid} -->|calls| target")

        for i, sym in enumerate(self.transitive_callers[:12]):
            if sym.id in direct_ids:
                continue
            nid = f"tc{i}"
            lines.append(f'    {nid}["{_lbl(sym)}"]')
            lines.append(f"    {nid} -.->|transitive| target")

        for i, sym in enumerate(self.subclasses[:6]):
            nid = f"sub{i}"
            lines.append(f'    {nid}["{_lbl(sym)}"]')
            lines.append(f"    {nid} -->|inherits| target")

        lines.append("    classDef changed fill:#f96,stroke:#c33,color:#000")
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
    decorators: list[str] = Field(default_factory=list)
    """Decorator / annotation names applied to this symbol (stripped of ``@``)."""


class RawInjection(BaseModel):
    """A dependency-injection edge extracted from constructor or field types.

    Represents a structural dependency where ``class_name`` depends on
    ``field_type`` via a typed constructor parameter or annotated class field.

    Examples
    --------
    Python constructor injection::

        class OrderService:
            def __init__(self, repo: OrderRepository): ...
        # → RawInjection(class_name="OrderService", field_name="repo",
        #                field_type="OrderRepository")

    Python field annotation::

        class PaymentService:
            gateway: StripeGateway
        # → RawInjection(class_name="PaymentService", field_name="gateway",
        #                field_type="StripeGateway")

    Java field annotation::

        @Autowired private UserRepository userRepo;
        # → RawInjection(class_name="<enclosing class>", field_name="userRepo",
        #                field_type="UserRepository")
    """

    class_name: str
    field_name: str | None = None
    field_type: str
    line: int = 0


class FileExtractionResult(BaseModel):
    """All raw graph data extracted from a single source file."""

    file_path: str
    language: Language
    symbols: list[RawSymbol] = Field(default_factory=list)
    imports: list[RawImport] = Field(default_factory=list)
    calls: list[RawCall] = Field(default_factory=list)
    inheritance: list[RawInheritance] = Field(default_factory=list)
    injections: list[RawInjection] = Field(default_factory=list)
    """Dependency-injection edges (constructor params / typed fields)."""
