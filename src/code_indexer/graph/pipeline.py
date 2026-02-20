"""Graph indexing pipeline.

This module orchestrates the full graph extraction and storage process:

1. Walk the codebase (reusing :class:`~code_indexer.indexer.walker.CodebaseWalker`).
2. For each file, run :class:`~code_indexer.graph.extractor.GraphExtractor` to
   get raw symbols, imports, calls, and inheritance.
3. Build :class:`~code_indexer.graph.models.FileNode`,
   :class:`~code_indexer.graph.models.SymbolNode`, and
   :class:`~code_indexer.graph.models.DirectoryNode` objects.
4. Resolve import strings to concrete file nodes using
   :class:`~code_indexer.graph.resolver.ImportResolver`.
5. Link call expressions to symbol nodes by name matching.
6. Assemble a :class:`~code_indexer.graph.models.GraphSnapshot` and upsert
   it into the configured :class:`~code_indexer.graph.base.BaseGraphStore`.

The pipeline is designed to run **alongside** (or after) the vector indexing
pipeline.  Both pipelines share the same file walker, so you can optionally
run them in tandem from the CLI or API.

Usage::

    from code_indexer.graph.pipeline import GraphIndexingPipeline, build_graph_store
    from code_indexer.core.config import get_settings

    settings = get_settings()
    pipeline = GraphIndexingPipeline(graph_store=build_graph_store(settings))
    result = pipeline.index_directory("./my-project")
    print(f"Indexed {result.symbols_created} symbols, {result.edges_created} edges")
"""

from __future__ import annotations

import logging
import posixpath
import time
from dataclasses import dataclass, field

from code_indexer.core.config import Settings, get_settings
from code_indexer.core.models import Language, SourceFile
from code_indexer.graph.base import BaseGraphStore
from code_indexer.graph.extractor import GraphExtractor
from code_indexer.graph.in_memory_graph import InMemoryGraphStore
from code_indexer.graph.models import (
    DirectoryNode,
    FileExtractionResult,
    FileNode,
    GraphEdge,
    GraphSnapshot,
    GraphStats,
    RelType,
    SymbolKind,
    SymbolNode,
)
from code_indexer.graph.resolver import ImportResolver
from code_indexer.indexer.walker import CodebaseWalker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class GraphIndexResult:
    """Summary of a graph indexing run."""

    files_processed: int = 0
    files_skipped: int = 0
    symbols_created: int = 0
    edges_created: int = 0
    elapsed_seconds: float = 0.0
    errors: list[tuple[str, str]] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        total = self.files_processed + self.files_skipped
        if total == 0:
            return 1.0
        return self.files_processed / total


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def build_graph_store(settings: Settings) -> BaseGraphStore:
    """Construct a :class:`BaseGraphStore` from application settings."""
    provider = settings.graph.provider if hasattr(settings, "graph") else "in_memory"

    if provider == "neo4j":
        from code_indexer.graph.neo4j_store import Neo4jGraphStore  # noqa: PLC0415

        gs = getattr(settings, "graph", None)
        return Neo4jGraphStore(
            uri=getattr(gs, "neo4j_uri", "bolt://localhost:7687"),
            username=getattr(gs, "neo4j_username", "neo4j"),
            password=getattr(gs, "neo4j_password", "password"),
            database=getattr(gs, "neo4j_database", "neo4j"),
        )
    else:
        return InMemoryGraphStore()


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class GraphIndexingPipeline:
    """Extracts and stores a code knowledge graph for a repository.

    Parameters
    ----------
    settings:
        Application settings (used to configure the file walker).
    graph_store:
        Target graph store.  Defaults to an :class:`InMemoryGraphStore`.
    extractor:
        :class:`GraphExtractor` instance.  Created automatically if not supplied.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        graph_store: BaseGraphStore | None = None,
        extractor: GraphExtractor | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._store = graph_store or InMemoryGraphStore()
        self._extractor = extractor or GraphExtractor()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def index_directory(
        self,
        path: str,
        *,
        extra_metadata: dict | None = None,
        clear_existing: bool = False,
    ) -> GraphIndexResult:
        """Index all source files under ``path``.

        Parameters
        ----------
        path:
            Root directory to walk.
        extra_metadata:
            Ignored (kept for API symmetry with :class:`IndexingPipeline`).
        clear_existing:
            If ``True``, wipe the graph store before indexing.
        """
        if clear_existing:
            self._store.clear()

        start = time.perf_counter()
        result = GraphIndexResult()

        walker = CodebaseWalker(
            root=path,
            ignore_patterns=self._settings.parser.ignore_patterns,
            max_file_size=self._settings.parser.max_file_size_bytes,
        )

        # --- Pass 1: extract raw data from every file ---
        per_file_results: list[FileExtractionResult] = []
        source_files: list[SourceFile] = []

        for source_file in walker.walk():
            try:
                extraction = self._extractor.extract_from_source(
                    file_path=source_file.path,
                    language=source_file.language,
                    content=source_file.content,
                )
                per_file_results.append(extraction)
                source_files.append(source_file)
                result.files_processed += 1
            except Exception as exc:  # noqa: BLE001
                result.files_skipped += 1
                result.errors.append((source_file.path, str(exc)))
                logger.warning("Graph extraction error for %s: %s", source_file.path, exc)

        # --- Pass 2: build graph snapshot ---
        snapshot = self._build_snapshot(source_files, per_file_results)
        result.symbols_created = len(snapshot.symbol_nodes)
        result.edges_created = len(snapshot.edges)

        # --- Pass 3: upsert into store ---
        self._store.upsert(snapshot)

        result.elapsed_seconds = time.perf_counter() - start
        logger.info(
            "Graph indexing complete: %d files, %d symbols, %d edges in %.1fs",
            result.files_processed,
            result.symbols_created,
            result.edges_created,
            result.elapsed_seconds,
        )
        return result

    def get_stats(self) -> GraphStats:
        return self._store.get_stats()

    @property
    def store(self) -> BaseGraphStore:
        return self._store

    # ------------------------------------------------------------------
    # Snapshot assembly
    # ------------------------------------------------------------------

    def _build_snapshot(
        self,
        source_files: list[SourceFile],
        per_file_results: list[FileExtractionResult],
    ) -> GraphSnapshot:
        """Assemble a :class:`GraphSnapshot` from per-file extraction results."""
        snapshot = GraphSnapshot()

        # Build file nodes and directory nodes
        for sf in source_files:
            fn = FileNode.from_path(sf.path, sf.language, sf.sha256)
            snapshot.file_nodes[fn.id] = fn

            # Directory hierarchy
            dir_path = posixpath.dirname(sf.path)
            while dir_path and dir_path != ".":
                dn = DirectoryNode.from_path(dir_path)
                if dn.id not in snapshot.directory_nodes:
                    snapshot.directory_nodes[dn.id] = dn
                parent_path = posixpath.dirname(dir_path)
                if parent_path and parent_path != dir_path:
                    parent_dn = DirectoryNode.from_path(parent_path)
                    snapshot.directory_nodes[parent_dn.id] = parent_dn
                    # PART_OF: dir → parent dir
                    e = GraphEdge.create(RelType.PART_OF, dn.id, parent_dn.id)
                    snapshot.edges.append(e)
                # PART_OF: file → dir
                e = GraphEdge.create(RelType.PART_OF, fn.id, dn.id)
                snapshot.edges.append(e)
                break  # Only one level; full tree built recursively above

        # Build file_map for resolver
        file_map = {fn.path: fn for fn in snapshot.file_nodes.values()}
        resolver = ImportResolver(file_map)

        # Build symbol nodes and edges per file
        for extraction in per_file_results:
            file_node = file_map.get(extraction.file_path)
            if file_node is None:
                continue

            # --- Symbol nodes ---
            # Map short name → SymbolNode for call resolution within the file
            local_symbols: dict[str, SymbolNode] = {}

            for raw in extraction.symbols:
                # Build qualified name: parent.name or just name
                if raw.parent_name:
                    qname = f"{file_node.path}::{raw.parent_name}.{raw.name}"
                else:
                    qname = f"{file_node.path}::{raw.name}"

                sn = SymbolNode(
                    id=SymbolNode.make_id(file_node.id, qname),
                    name=raw.name,
                    qualified_name=qname,
                    kind=raw.kind,
                    file_id=file_node.id,
                    file_path=file_node.path,
                    start_line=raw.start_line,
                    end_line=raw.end_line,
                    parent_name=raw.parent_name,
                )
                snapshot.symbol_nodes[sn.id] = sn
                local_symbols[raw.name] = sn

                # DEFINES: file → symbol
                snapshot.edges.append(
                    GraphEdge.create(RelType.DEFINES, file_node.id, sn.id)
                )

            # CONTAINS: class → method
            for raw in extraction.symbols:
                if raw.parent_name and raw.parent_name in local_symbols:
                    parent_sn = local_symbols[raw.parent_name]
                    child_qname = f"{file_node.path}::{raw.parent_name}.{raw.name}"
                    child_id = SymbolNode.make_id(file_node.id, child_qname)
                    if child_id in snapshot.symbol_nodes:
                        snapshot.edges.append(
                            GraphEdge.create(RelType.CONTAINS, parent_sn.id, child_id)
                        )

            # --- Import edges ---
            for raw_import in extraction.imports:
                target_file = resolver.resolve(
                    raw_import.module_string,
                    extraction.file_path,
                    extraction.language,
                )
                target_id = target_file.id if target_file else f"__ext__{raw_import.module_string}"
                # For external imports, we don't create a file node but still record the edge
                # on the source file with metadata
                snapshot.edges.append(
                    GraphEdge.create(
                        RelType.IMPORTS,
                        file_node.id,
                        target_id if target_file else file_node.id,  # self-loop marker for external
                        module_string=raw_import.module_string,
                        imported_names=raw_import.imported_names,
                        alias=raw_import.alias,
                        is_relative=raw_import.is_relative,
                        is_external=target_file is None,
                        line=raw_import.line,
                    )
                )

            # --- Inheritance edges ---
            for raw_inh in extraction.inheritance:
                class_sn = local_symbols.get(raw_inh.class_name)
                if class_sn is None:
                    continue
                for base_name in raw_inh.base_names:
                    # Try to find the base class: first in same file, then globally
                    base_sn = local_symbols.get(base_name)
                    if base_sn:
                        snapshot.edges.append(
                            GraphEdge.create(
                                RelType.INHERITS_FROM,
                                class_sn.id,
                                base_sn.id,
                                line=raw_inh.line,
                            )
                        )

        # --- Cross-file call resolution (second pass) ---
        # Build global name index: short name → [SymbolNode]
        global_name_index: dict[str, list[SymbolNode]] = {}
        for sn in snapshot.symbol_nodes.values():
            global_name_index.setdefault(sn.name, []).append(sn)

        # Now link calls within each file
        for extraction in per_file_results:
            file_node = file_map.get(extraction.file_path)
            if file_node is None:
                continue

            # Local symbols for this file
            local_syms_by_name: dict[str, SymbolNode] = {
                sn.name: sn
                for sn in snapshot.symbol_nodes.values()
                if sn.file_id == file_node.id
            }

            for raw_call in extraction.calls:
                if not raw_call.callee_name:
                    continue

                # Find caller: a function that contains this line number
                caller_sn = self._find_enclosing_symbol(
                    raw_call.line, file_node.id, snapshot
                )
                if caller_sn is None:
                    continue

                # Resolve callee: prefer local file, fall back to global
                callee_sn = local_syms_by_name.get(raw_call.callee_name)
                if callee_sn is None:
                    candidates = global_name_index.get(raw_call.callee_name, [])
                    # Pick the first candidate (ambiguous); could be refined with type info
                    callee_sn = candidates[0] if candidates else None

                if callee_sn and callee_sn.id != caller_sn.id:
                    snapshot.edges.append(
                        GraphEdge.create(
                            RelType.CALLS,
                            caller_sn.id,
                            callee_sn.id,
                            line=raw_call.line,
                        )
                    )

        # --- Cross-file inheritance resolution ---
        for extraction in per_file_results:
            file_node = file_map.get(extraction.file_path)
            if file_node is None:
                continue

            local_syms = {
                sn.name: sn
                for sn in snapshot.symbol_nodes.values()
                if sn.file_id == file_node.id
            }

            for raw_inh in extraction.inheritance:
                class_sn = local_syms.get(raw_inh.class_name)
                if class_sn is None:
                    continue
                for base_name in raw_inh.base_names:
                    # Only add cross-file edges (intra-file already done above)
                    if base_name in local_syms:
                        continue
                    candidates = global_name_index.get(base_name, [])
                    if candidates:
                        base_sn = candidates[0]
                        # Check not already added
                        snapshot.edges.append(
                            GraphEdge.create(
                                RelType.INHERITS_FROM,
                                class_sn.id,
                                base_sn.id,
                                line=raw_inh.line,
                            )
                        )

        return snapshot

    @staticmethod
    def _find_enclosing_symbol(
        line: int, file_id: str, snapshot: GraphSnapshot
    ) -> SymbolNode | None:
        """Find the smallest symbol in ``file_id`` that contains ``line``."""
        best: SymbolNode | None = None
        best_size = float("inf")
        for sn in snapshot.symbol_nodes.values():
            if sn.file_id != file_id:
                continue
            if sn.start_line <= line <= sn.end_line:
                size = sn.end_line - sn.start_line
                if size < best_size:
                    best = sn
                    best_size = size
        return best
