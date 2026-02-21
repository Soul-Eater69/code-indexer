"""Abstract base class for code knowledge graph stores."""

from __future__ import annotations

from abc import ABC, abstractmethod

from code_indexer.graph.models import (
    GraphSnapshot,
    GraphStats,
    ImpactResult,
    SymbolContext,
    SymbolNode,
)


class BaseGraphStore(ABC):
    """Abstract interface for code knowledge graph storage.

    Implementations must support:
    * Bulk upsert of a full :class:`~code_indexer.graph.models.GraphSnapshot`.
    * Symbol lookup by name / qualified name / file.
    * Graph traversal queries (callers, callees, import graph, inheritance).
    * Shortest-path queries between two symbols.
    * Statistics and bulk clear.

    All methods should be **idempotent**: re-indexing the same content
    should not create duplicate nodes or edges.
    """

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    @abstractmethod
    def upsert(self, snapshot: GraphSnapshot) -> None:
        """Merge a :class:`GraphSnapshot` into the graph.

        Nodes and edges that already exist (identified by their stable
        deterministic IDs) should be updated rather than duplicated.
        """

    @abstractmethod
    def delete_file(self, file_id: str) -> int:
        """Remove all nodes and edges associated with ``file_id``.

        Returns the number of nodes/edges removed.
        """

    @abstractmethod
    def clear(self) -> None:
        """Wipe the entire graph store."""

    # ------------------------------------------------------------------
    # Node lookup
    # ------------------------------------------------------------------

    @abstractmethod
    def get_symbol_by_id(self, symbol_id: str) -> SymbolNode | None:
        """Return a :class:`SymbolNode` by its stable ID, or ``None``."""

    @abstractmethod
    def find_symbols_by_name(
        self, name: str, *, kind: str | None = None, file_path: str | None = None
    ) -> list[SymbolNode]:
        """Return all symbols whose short ``name`` matches ``name``.

        Optional filters:
        * ``kind``      — filter by :class:`~code_indexer.graph.models.SymbolKind` value.
        * ``file_path`` — filter by file path (prefix or exact match).
        """

    @abstractmethod
    def find_symbols_in_file(self, file_id: str) -> list[SymbolNode]:
        """Return all symbols defined in ``file_id``."""

    # ------------------------------------------------------------------
    # Traversal queries
    # ------------------------------------------------------------------

    @abstractmethod
    def get_callers(self, symbol_id: str, depth: int = 1) -> list[SymbolNode]:
        """Return symbols that directly call ``symbol_id``.

        When ``depth > 1``, return all transitive callers up to ``depth``
        hops away.
        """

    @abstractmethod
    def get_callees(self, symbol_id: str, depth: int = 1) -> list[SymbolNode]:
        """Return symbols that ``symbol_id`` calls.

        When ``depth > 1``, return all transitive callees up to ``depth``
        hops away.
        """

    @abstractmethod
    def get_call_path(
        self, from_symbol_id: str, to_symbol_id: str
    ) -> list[SymbolNode] | None:
        """Find the shortest call path from one symbol to another.

        Returns an ordered list of symbols forming the path, or ``None``
        if no path exists.
        """

    @abstractmethod
    def get_import_graph(self, file_id: str) -> dict[str, list[str]]:
        """Return the import adjacency map for files reachable from ``file_id``.

        Returns ``{file_path: [imported_file_path, ...]}`` for the subgraph
        rooted at ``file_id``.
        """

    @abstractmethod
    def get_subclasses(self, symbol_id: str) -> list[SymbolNode]:
        """Return direct subclasses / implementors of ``symbol_id``."""

    @abstractmethod
    def get_superclasses(self, symbol_id: str) -> list[SymbolNode]:
        """Return direct base classes / interfaces of ``symbol_id``."""

    @abstractmethod
    def get_symbol_context(self, symbol_id: str) -> SymbolContext | None:
        """Return full structural context for ``symbol_id``.

        This is the primary query for RAG usage: returns callers, callees,
        parent class, methods (if a class), inheritance chain, and import graph.
        """

    def get_impact_set(
        self, symbol_id: str, *, min_confidence: float = 0.0
    ) -> ImpactResult | None:
        """Return the full impact set for a symbol change.

        Combines three impact dimensions:

        1. **Call-graph reverse reachability** — all transitive callers,
           filtered by the ``min_confidence`` threshold on each edge.
        2. **Inheritance reverse** — direct subclasses / implementors.
        3. **File-level reverse imports** — files that directly import
           the file defining this symbol.

        Parameters
        ----------
        symbol_id:
            Stable ID of the symbol that is being changed.
        min_confidence:
            Minimum compound edge confidence to follow (0.0 = all edges,
            0.85 = certain-only, 0.50 = probable + certain).

        Returns ``None`` if ``symbol_id`` is not found.

        Concrete stores should override this for confidence-aware traversal.
        The default implementation uses existing traversal methods at a fixed
        large depth with no compound-confidence tracking.
        """
        sn = self.get_symbol_by_id(symbol_id)
        if sn is None:
            return None

        ctx = self.get_symbol_context(symbol_id)
        if ctx is None:
            return None

        direct_callers = self.get_callers(symbol_id, depth=1)
        transitive_callers = self.get_callers(symbol_id, depth=100)
        subclasses = self.get_subclasses(symbol_id)

        affected_file_ids = {s.file_id for s in transitive_callers} | {
            s.file_id for s in subclasses
        }
        # Build FileNode list from symbol context's file_imported_by (available at 1 hop)
        importing_files = ctx.file_imported_by

        return ImpactResult(
            symbol=sn,
            direct_callers=direct_callers,
            transitive_callers=transitive_callers,
            subclasses=subclasses,
            importing_files=importing_files,
            affected_files=[],  # concrete stores provide FileNode objects
            confidence_breakdown={"certain": len(transitive_callers)},
            min_confidence_used=min_confidence,
        )

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    @abstractmethod
    def get_stats(self) -> GraphStats:
        """Return summary statistics about the current graph."""

    # ------------------------------------------------------------------
    # Persistence (optional — only in-memory stores need this)
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Serialise the graph to disk at ``path``.

        Default implementation raises :exc:`NotImplementedError`.
        Override in stores that support on-disk persistence.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support save()")

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable backend name."""
