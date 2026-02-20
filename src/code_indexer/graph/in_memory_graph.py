"""In-memory code knowledge graph store built on NetworkX.

Uses a ``networkx.MultiDiGraph`` where:
* Every node has a ``data`` attribute holding the serialised
  :class:`~code_indexer.graph.models.FileNode`, :class:`SymbolNode`, or
  :class:`DirectoryNode`.
* Every edge has ``rel_type`` and ``properties`` attributes.

The store is suitable for:
* Unit tests (zero infrastructure dependencies).
* Development / quick exploration.
* Codebases with < 100k nodes (NetworkX graph fits comfortably in RAM).

For production use, switch to
:class:`~code_indexer.graph.neo4j_store.Neo4jGraphStore`.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any

from code_indexer.graph.base import BaseGraphStore
from code_indexer.graph.models import (
    DirectoryNode,
    FileNode,
    GraphEdge,
    GraphSnapshot,
    GraphStats,
    NodeType,
    RelType,
    SymbolContext,
    SymbolNode,
)

logger = logging.getLogger(__name__)

try:
    import networkx as nx  # type: ignore[import-untyped]

    _NX_AVAILABLE = True
except ImportError:
    _NX_AVAILABLE = False
    logger.warning(
        "networkx is not installed — InMemoryGraphStore will use a simple dict fallback. "
        "Install with: pip install networkx"
    )


class _DictFallbackGraph:
    """Minimal adjacency-list graph used when networkx is not available."""

    def __init__(self) -> None:
        self._nodes: dict[str, dict[str, Any]] = {}
        self._out: dict[str, list[dict[str, Any]]] = {}
        self._in: dict[str, list[dict[str, Any]]] = {}

    def add_node(self, node_id: str, **attrs: Any) -> None:
        self._nodes[node_id] = attrs
        self._out.setdefault(node_id, [])
        self._in.setdefault(node_id, [])

    def has_node(self, node_id: str) -> bool:
        return node_id in self._nodes

    def add_edge(self, src: str, tgt: str, **attrs: Any) -> None:
        self._out.setdefault(src, []).append({"target": tgt, **attrs})
        self._in.setdefault(tgt, []).append({"source": src, **attrs})

    def node_data(self, node_id: str) -> dict[str, Any]:
        return self._nodes.get(node_id, {})

    def out_edges(self, node_id: str, rel_type: RelType | None = None) -> list[dict[str, Any]]:
        edges = self._out.get(node_id, [])
        if rel_type:
            edges = [e for e in edges if e.get("rel_type") == rel_type]
        return edges

    def in_edges(self, node_id: str, rel_type: RelType | None = None) -> list[dict[str, Any]]:
        edges = self._in.get(node_id, [])
        if rel_type:
            edges = [e for e in edges if e.get("rel_type") == rel_type]
        return edges

    def nodes(self) -> dict[str, dict[str, Any]]:
        return self._nodes

    def clear(self) -> None:
        self._nodes.clear()
        self._out.clear()
        self._in.clear()

    def __len__(self) -> int:
        return len(self._nodes)


class InMemoryGraphStore(BaseGraphStore):
    """In-memory code knowledge graph backed by NetworkX (or a dict fallback).

    Parameters
    ----------
    use_networkx:
        If ``True`` (default), use NetworkX for graph traversal (BFS, shortest
        path).  Falls back automatically if networkx is not installed.
    """

    def __init__(self, *, use_networkx: bool = True) -> None:
        if use_networkx and _NX_AVAILABLE:
            self._g: Any = nx.MultiDiGraph()
            self._using_nx = True
        else:
            self._g = _DictFallbackGraph()
            self._using_nx = False

        # Secondary indexes for fast lookup
        self._symbols_by_name: dict[str, list[str]] = {}  # name → [symbol_id]
        self._symbols_by_file: dict[str, list[str]] = {}  # file_id → [symbol_id]
        self._files_by_path: dict[str, str] = {}          # path → file_id
        self._file_ids: set[str] = set()

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def upsert(self, snapshot: GraphSnapshot) -> None:
        """Merge a :class:`GraphSnapshot` into the in-memory graph."""
        # Add file nodes
        for node_id, fn in snapshot.file_nodes.items():
            self._add_node(node_id, data=fn, node_type=NodeType.FILE)
            self._files_by_path[fn.path] = node_id
            self._file_ids.add(node_id)

        # Add symbol nodes
        for node_id, sn in snapshot.symbol_nodes.items():
            self._add_node(node_id, data=sn, node_type=NodeType.SYMBOL)
            self._symbols_by_name.setdefault(sn.name, [])
            if node_id not in self._symbols_by_name[sn.name]:
                self._symbols_by_name[sn.name].append(node_id)
            self._symbols_by_file.setdefault(sn.file_id, [])
            if node_id not in self._symbols_by_file[sn.file_id]:
                self._symbols_by_file[sn.file_id].append(node_id)

        # Add directory nodes
        for node_id, dn in snapshot.directory_nodes.items():
            self._add_node(node_id, data=dn, node_type=NodeType.DIRECTORY)

        # Add edges
        for edge in snapshot.edges:
            self._add_edge(edge)

        logger.debug(
            "Graph upsert: %d files, %d symbols, %d edges",
            len(snapshot.file_nodes),
            len(snapshot.symbol_nodes),
            len(snapshot.edges),
        )

    def delete_file(self, file_id: str) -> int:
        """Remove all nodes and edges for ``file_id``."""
        removed = 0

        # Remove symbol nodes defined in this file
        symbol_ids = list(self._symbols_by_file.pop(file_id, []))
        for sid in symbol_ids:
            self._remove_node(sid)
            removed += 1
            # Clean secondary index
            sn = self._get_symbol(sid)
            if sn and sn.name in self._symbols_by_name:
                try:
                    self._symbols_by_name[sn.name].remove(sid)
                except ValueError:
                    pass

        # Remove file node itself
        if self._has_node(file_id):
            fn = self._get_file(file_id)
            if fn:
                self._files_by_path.pop(fn.path, None)
            self._remove_node(file_id)
            self._file_ids.discard(file_id)
            removed += 1

        return removed

    def clear(self) -> None:
        if self._using_nx:
            self._g.clear()
        else:
            self._g.clear()
        self._symbols_by_name.clear()
        self._symbols_by_file.clear()
        self._files_by_path.clear()
        self._file_ids.clear()

    # ------------------------------------------------------------------
    # Node lookup
    # ------------------------------------------------------------------

    def get_symbol_by_id(self, symbol_id: str) -> SymbolNode | None:
        return self._get_symbol(symbol_id)

    def find_symbols_by_name(
        self, name: str, *, kind: str | None = None, file_path: str | None = None
    ) -> list[SymbolNode]:
        ids = self._symbols_by_name.get(name, [])
        result: list[SymbolNode] = []
        for sid in ids:
            sn = self._get_symbol(sid)
            if sn is None:
                continue
            if kind and sn.kind.value != kind:
                continue
            if file_path and not sn.file_path.startswith(file_path):
                continue
            result.append(sn)
        return result

    def find_symbols_in_file(self, file_id: str) -> list[SymbolNode]:
        ids = self._symbols_by_file.get(file_id, [])
        return [s for sid in ids if (s := self._get_symbol(sid)) is not None]

    # ------------------------------------------------------------------
    # Traversal
    # ------------------------------------------------------------------

    def get_callers(self, symbol_id: str, depth: int = 1) -> list[SymbolNode]:
        return self._bfs_incoming(symbol_id, RelType.CALLS, depth)

    def get_callees(self, symbol_id: str, depth: int = 1) -> list[SymbolNode]:
        return self._bfs_outgoing(symbol_id, RelType.CALLS, depth)

    def get_call_path(
        self, from_symbol_id: str, to_symbol_id: str
    ) -> list[SymbolNode] | None:
        if self._using_nx:
            return self._nx_shortest_path(from_symbol_id, to_symbol_id, RelType.CALLS)
        return self._bfs_path(from_symbol_id, to_symbol_id, RelType.CALLS)

    def get_import_graph(self, file_id: str) -> dict[str, list[str]]:
        """Return ``{file_path: [imported_file_path]}`` BFS from ``file_id``."""
        result: dict[str, list[str]] = {}
        visited: set[str] = set()
        queue = deque([file_id])
        while queue:
            fid = queue.popleft()
            if fid in visited:
                continue
            visited.add(fid)
            fn = self._get_file(fid)
            if fn is None:
                continue
            targets = self._out_neighbours(fid, RelType.IMPORTS)
            target_paths = []
            for tid in targets:
                tf = self._get_file(tid)
                if tf:
                    target_paths.append(tf.path)
                    queue.append(tid)
            result[fn.path] = target_paths
        return result

    def get_subclasses(self, symbol_id: str) -> list[SymbolNode]:
        return self._in_neighbours_symbols(symbol_id, RelType.INHERITS_FROM)

    def get_superclasses(self, symbol_id: str) -> list[SymbolNode]:
        return self._out_neighbours_symbols(symbol_id, RelType.INHERITS_FROM)

    def get_symbol_context(self, symbol_id: str) -> SymbolContext | None:
        sn = self._get_symbol(symbol_id)
        if sn is None:
            return None

        fn = self._get_file(sn.file_id)
        if fn is None:
            return None

        # Parent class (via CONTAINS edge inbound)
        parent_class: SymbolNode | None = None
        for pid in self._in_neighbours(symbol_id, RelType.CONTAINS):
            parent_class = self._get_symbol(pid)
            break

        # Methods (via CONTAINS edge outbound) — only if this symbol is a class
        methods = self._out_neighbours_symbols(symbol_id, RelType.CONTAINS)

        # Call graph
        callers = self._bfs_incoming(symbol_id, RelType.CALLS, depth=1)
        callees = self._bfs_outgoing(symbol_id, RelType.CALLS, depth=1)

        # Inheritance
        inherits_from = self.get_superclasses(symbol_id)
        subclasses = self.get_subclasses(symbol_id)

        # File-level imports/imported-by
        file_import_ids = self._out_neighbours(sn.file_id, RelType.IMPORTS)
        file_imports = [f for fid in file_import_ids if (f := self._get_file(fid)) is not None]
        file_imported_by_ids = self._in_neighbours(sn.file_id, RelType.IMPORTS)
        file_imported_by = [
            f for fid in file_imported_by_ids if (f := self._get_file(fid)) is not None
        ]

        return SymbolContext(
            symbol=sn,
            file=fn,
            callers=callers,
            callees=callees,
            parent_class=parent_class,
            methods=methods,
            inherits_from=inherits_from,
            subclasses=subclasses,
            file_imports=file_imports,
            file_imported_by=file_imported_by,
        )

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_stats(self) -> GraphStats:
        from collections import Counter

        symbols = [self._get_symbol(sid) for sid in self._get_all_symbol_ids()]
        symbols = [s for s in symbols if s is not None]
        files = [self._get_file(fid) for fid in self._file_ids]
        files = [f for f in files if f is not None]

        edges: list[GraphEdge] = []
        if self._using_nx:
            for _src, _tgt, data in self._g.edges(data=True):
                if "edge" in data:
                    edges.append(data["edge"])
        else:
            for _, edge_list in self._g._out.items():
                for ed in edge_list:
                    if "edge" in ed:
                        edges.append(ed["edge"])

        edges_by_type = dict(Counter(e.rel_type.value for e in edges))
        symbols_by_kind = dict(Counter(s.kind.value for s in symbols))
        languages = dict(Counter(f.language.value for f in files))
        dir_count = sum(
            1
            for nid, data in (
                self._g.nodes(data=True) if self._using_nx else self._g.nodes().items()
            )
            if (data.get("node_type") if self._using_nx else data.get("node_type")) == NodeType.DIRECTORY
        )

        return GraphStats(
            total_files=len(files),
            total_symbols=len(symbols),
            total_directories=dir_count,
            total_edges=len(edges),
            edges_by_type=edges_by_type,
            symbols_by_kind=symbols_by_kind,
            languages=languages,
            graph_store=self.name,
        )

    @property
    def name(self) -> str:
        if self._using_nx:
            return "InMemoryGraph (NetworkX)"
        return "InMemoryGraph (dict fallback)"

    # ------------------------------------------------------------------
    # Internal helpers — NetworkX vs dict fallback abstraction
    # ------------------------------------------------------------------

    def _add_node(self, node_id: str, **attrs: Any) -> None:
        if self._using_nx:
            self._g.add_node(node_id, **attrs)
        else:
            self._g.add_node(node_id, **attrs)

    def _has_node(self, node_id: str) -> bool:
        if self._using_nx:
            return self._g.has_node(node_id)
        return self._g.has_node(node_id)

    def _remove_node(self, node_id: str) -> None:
        if self._using_nx:
            if self._g.has_node(node_id):
                self._g.remove_node(node_id)
        else:
            self._g._nodes.pop(node_id, None)
            self._g._out.pop(node_id, None)
            # Remove from other nodes' in-edge lists
            for n in self._g._in:
                self._g._in[n] = [e for e in self._g._in[n] if e.get("source") != node_id]
            self._g._in.pop(node_id, None)

    def _add_edge(self, edge: GraphEdge) -> None:
        if self._using_nx:
            self._g.add_edge(
                edge.source_id,
                edge.target_id,
                key=edge.id,
                rel_type=edge.rel_type,
                properties=edge.properties,
                edge=edge,
            )
        else:
            self._g.add_edge(
                edge.source_id,
                edge.target_id,
                rel_type=edge.rel_type,
                properties=edge.properties,
                edge=edge,
            )

    def _get_symbol(self, node_id: str) -> SymbolNode | None:
        if self._using_nx:
            if not self._g.has_node(node_id):
                return None
            data = self._g.nodes[node_id].get("data")
        else:
            data = self._g.node_data(node_id).get("data")
        if isinstance(data, SymbolNode):
            return data
        return None

    def _get_file(self, node_id: str) -> FileNode | None:
        if self._using_nx:
            if not self._g.has_node(node_id):
                return None
            data = self._g.nodes[node_id].get("data")
        else:
            data = self._g.node_data(node_id).get("data")
        if isinstance(data, FileNode):
            return data
        return None

    def _get_all_symbol_ids(self) -> list[str]:
        ids = []
        for sid_list in self._symbols_by_file.values():
            ids.extend(sid_list)
        return ids

    def _out_neighbours(self, node_id: str, rel_type: RelType) -> list[str]:
        if self._using_nx:
            result = []
            for _src, tgt, data in self._g.out_edges(node_id, data=True):
                if data.get("rel_type") == rel_type:
                    result.append(tgt)
            return result
        return [e["target"] for e in self._g.out_edges(node_id, rel_type)]

    def _in_neighbours(self, node_id: str, rel_type: RelType) -> list[str]:
        if self._using_nx:
            result = []
            for src, _tgt, data in self._g.in_edges(node_id, data=True):
                if data.get("rel_type") == rel_type:
                    result.append(src)
            return result
        return [e["source"] for e in self._g.in_edges(node_id, rel_type)]

    def _out_neighbours_symbols(self, node_id: str, rel_type: RelType) -> list[SymbolNode]:
        return [s for nid in self._out_neighbours(node_id, rel_type) if (s := self._get_symbol(nid)) is not None]

    def _in_neighbours_symbols(self, node_id: str, rel_type: RelType) -> list[SymbolNode]:
        return [s for nid in self._in_neighbours(node_id, rel_type) if (s := self._get_symbol(nid)) is not None]

    def _bfs_outgoing(
        self, start_id: str, rel_type: RelType, depth: int
    ) -> list[SymbolNode]:
        visited: set[str] = {start_id}
        result: list[SymbolNode] = []
        queue: deque[tuple[str, int]] = deque([(start_id, 0)])
        while queue:
            node_id, d = queue.popleft()
            if d >= depth:
                continue
            for nid in self._out_neighbours(node_id, rel_type):
                if nid in visited:
                    continue
                visited.add(nid)
                sn = self._get_symbol(nid)
                if sn:
                    result.append(sn)
                queue.append((nid, d + 1))
        return result

    def _bfs_incoming(
        self, start_id: str, rel_type: RelType, depth: int
    ) -> list[SymbolNode]:
        visited: set[str] = {start_id}
        result: list[SymbolNode] = []
        queue: deque[tuple[str, int]] = deque([(start_id, 0)])
        while queue:
            node_id, d = queue.popleft()
            if d >= depth:
                continue
            for nid in self._in_neighbours(node_id, rel_type):
                if nid in visited:
                    continue
                visited.add(nid)
                sn = self._get_symbol(nid)
                if sn:
                    result.append(sn)
                queue.append((nid, d + 1))
        return result

    def _bfs_path(
        self, from_id: str, to_id: str, rel_type: RelType
    ) -> list[SymbolNode] | None:
        """BFS shortest path (fallback when NetworkX is unavailable)."""
        if from_id == to_id:
            sn = self._get_symbol(from_id)
            return [sn] if sn else []
        visited: set[str] = {from_id}
        queue: deque[tuple[str, list[str]]] = deque([(from_id, [from_id])])
        while queue:
            node_id, path = queue.popleft()
            for nid in self._out_neighbours(node_id, rel_type):
                if nid in visited:
                    continue
                new_path = path + [nid]
                if nid == to_id:
                    symbols = [self._get_symbol(n) for n in new_path]
                    return [s for s in symbols if s is not None]
                visited.add(nid)
                queue.append((nid, new_path))
        return None

    def _nx_shortest_path(
        self, from_id: str, to_id: str, rel_type: RelType
    ) -> list[SymbolNode] | None:
        # Create a filtered view with only CALLS edges
        edges_to_use = [
            (s, t) for s, t, data in self._g.edges(data=True)
            if data.get("rel_type") == rel_type
        ]
        subgraph = nx.DiGraph()
        subgraph.add_edges_from(edges_to_use)
        try:
            path = nx.shortest_path(subgraph, from_id, to_id)
            return [s for nid in path if (s := self._get_symbol(nid)) is not None]
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None
