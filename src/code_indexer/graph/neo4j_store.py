"""Neo4j graph store for the code knowledge graph.

Production-grade persistent graph database with Cypher traversal support.

Neo4j Schema
------------
Nodes::

    (:File       {id, path, language, sha256})
    (:Symbol     {id, name, qualified_name, kind, file_id, file_path,
                  start_line, end_line, parent_name, docstring, decorators})
    (:Directory  {id, path, name})

Relationships::

    (:File)-[:IMPORTS     {module_string, imported_names, alias, is_relative}]->(:File)
    (:File)-[:DEFINES                                                          ]->(:Symbol)
    (:Symbol)-[:CONTAINS                                                       ]->(:Symbol)
    (:Symbol)-[:CALLS     {line, count}                                        ]->(:Symbol)
    (:Symbol)-[:INHERITS_FROM                                                  ]->(:Symbol)
    (:Symbol)-[:INJECTS   {field_name, field_type, line}                       ]->(:Symbol)
    (:Symbol)-[:REFERENCES {line}                                              ]->(:Symbol)
    (:File/:Directory)-[:PART_OF                                               ]->(:Directory)

Indexes created on startup::

    CREATE INDEX file_path_idx   FOR (f:File)   ON (f.path)
    CREATE INDEX symbol_name_idx FOR (s:Symbol) ON (s.name)
    CREATE INDEX symbol_qname    FOR (s:Symbol) ON (s.qualified_name)
    CREATE INDEX symbol_file_idx FOR (s:Symbol) ON (s.file_id)

Requires
--------
* A running Neo4j instance (Neo4j 5+, Community or Enterprise).
* ``pip install neo4j`` (added in the ``graph`` optional extras).
"""

from __future__ import annotations

import logging
from typing import Any

from code_indexer.graph.base import BaseGraphStore
from code_indexer.graph.models import (
    FileNode,
    GraphSnapshot,
    GraphStats,
    Language,
    RelType,
    SymbolContext,
    SymbolKind,
    SymbolNode,
)

logger = logging.getLogger(__name__)


class Neo4jGraphStore(BaseGraphStore):
    """Code knowledge graph store backed by Neo4j.

    Parameters
    ----------
    uri:
        Bolt URI of the Neo4j instance, e.g. ``"bolt://localhost:7687"``.
    username:
        Neo4j username (default ``"neo4j"``).
    password:
        Neo4j password.
    database:
        Target database name (default ``"neo4j"``).
    batch_size:
        How many nodes/edges to merge in a single Cypher transaction.
        Larger batches are faster but use more memory.
    """

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        username: str = "neo4j",
        password: str = "password",
        database: str = "neo4j",
        batch_size: int = 500,
    ) -> None:
        try:
            from neo4j import GraphDatabase  # type: ignore[import-untyped]  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "neo4j driver is not installed. Install it with: pip install neo4j"
            ) from exc

        self._driver = GraphDatabase.driver(uri, auth=(username, password))
        self._database = database
        self._batch_size = batch_size
        self._ensure_schema()

    # ------------------------------------------------------------------
    # Schema setup
    # ------------------------------------------------------------------

    def _ensure_schema(self) -> None:
        """Create indexes and constraints if they don't exist yet."""
        statements = [
            "CREATE INDEX file_id_idx IF NOT EXISTS FOR (f:File) ON (f.id)",
            "CREATE INDEX file_path_idx IF NOT EXISTS FOR (f:File) ON (f.path)",
            "CREATE INDEX symbol_id_idx IF NOT EXISTS FOR (s:Symbol) ON (s.id)",
            "CREATE INDEX symbol_name_idx IF NOT EXISTS FOR (s:Symbol) ON (s.name)",
            "CREATE INDEX symbol_qname_idx IF NOT EXISTS FOR (s:Symbol) ON (s.qualified_name)",
            "CREATE INDEX symbol_file_idx IF NOT EXISTS FOR (s:Symbol) ON (s.file_id)",
            "CREATE INDEX dir_path_idx IF NOT EXISTS FOR (d:Directory) ON (d.path)",
        ]
        with self._session() as session:
            for stmt in statements:
                session.run(stmt)
        logger.debug("Neo4j schema ensured")

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def upsert(self, snapshot: GraphSnapshot) -> None:
        """Merge a :class:`GraphSnapshot` into Neo4j."""
        # Upsert file nodes
        file_rows = [
            {
                "id": fn.id,
                "path": fn.path,
                "language": fn.language.value,
                "sha256": fn.sha256,
            }
            for fn in snapshot.file_nodes.values()
        ]
        self._batch_run(
            """
            UNWIND $rows AS row
            MERGE (f:File {id: row.id})
            SET f.path = row.path,
                f.language = row.language,
                f.sha256 = row.sha256
            """,
            file_rows,
        )

        # Upsert symbol nodes
        sym_rows = [
            {
                "id": sn.id,
                "name": sn.name,
                "qualified_name": sn.qualified_name,
                "kind": sn.kind.value,
                "file_id": sn.file_id,
                "file_path": sn.file_path,
                "start_line": sn.start_line,
                "end_line": sn.end_line,
                "parent_name": sn.parent_name,
                "docstring": sn.docstring,
                "decorators": sn.decorators,
            }
            for sn in snapshot.symbol_nodes.values()
        ]
        self._batch_run(
            """
            UNWIND $rows AS row
            MERGE (s:Symbol {id: row.id})
            SET s.name = row.name,
                s.qualified_name = row.qualified_name,
                s.kind = row.kind,
                s.file_id = row.file_id,
                s.file_path = row.file_path,
                s.start_line = row.start_line,
                s.end_line = row.end_line,
                s.parent_name = row.parent_name,
                s.docstring = row.docstring,
                s.decorators = row.decorators
            """,
            sym_rows,
        )

        # Upsert directory nodes
        dir_rows = [
            {"id": dn.id, "path": dn.path, "name": dn.name}
            for dn in snapshot.directory_nodes.values()
        ]
        self._batch_run(
            """
            UNWIND $rows AS row
            MERGE (d:Directory {id: row.id})
            SET d.path = row.path, d.name = row.name
            """,
            dir_rows,
        )

        # Group edges by type for efficient Cypher
        edges_by_type: dict[RelType, list[GraphEdge]] = {}  # type: ignore[name-defined]
        for edge in snapshot.edges:
            edges_by_type.setdefault(edge.rel_type, []).append(edge)

        for rel_type, edges in edges_by_type.items():
            self._upsert_edges(rel_type, edges)

        logger.info(
            "Neo4j upsert: %d files, %d symbols, %d edges",
            len(snapshot.file_nodes),
            len(snapshot.symbol_nodes),
            len(snapshot.edges),
        )

    def _upsert_edges(self, rel_type: RelType, edges: list[Any]) -> None:
        """Upsert a batch of edges of the same type."""
        rows = [
            {
                "id": e.id,
                "source_id": e.source_id,
                "target_id": e.target_id,
                "props": e.properties,
            }
            for e in edges
        ]
        # Use dynamic relationship type via APOC or string concatenation
        # We use a separate Cypher per rel_type for clarity/safety
        cypher = f"""
            UNWIND $rows AS row
            MATCH (src {{id: row.source_id}})
            MATCH (tgt {{id: row.target_id}})
            MERGE (src)-[r:{rel_type.value} {{id: row.id}}]->(tgt)
            SET r += row.props
        """
        self._batch_run(cypher, rows)

    def delete_file(self, file_id: str) -> int:
        cypher = """
            MATCH (f:File {id: $file_id})
            OPTIONAL MATCH (s:Symbol {file_id: $file_id})
            DETACH DELETE f, s
            RETURN count(f) + count(s) AS removed
        """
        with self._session() as session:
            result = session.run(cypher, file_id=file_id)
            record = result.single()
            return int(record["removed"]) if record else 0

    def clear(self) -> None:
        with self._session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        logger.info("Neo4j graph cleared")

    # ------------------------------------------------------------------
    # Node lookup
    # ------------------------------------------------------------------

    def get_symbol_by_id(self, symbol_id: str) -> SymbolNode | None:
        cypher = "MATCH (s:Symbol {id: $id}) RETURN s"
        with self._session() as session:
            result = session.run(cypher, id=symbol_id)
            record = result.single()
            if record:
                return self._record_to_symbol(record["s"])
        return None

    def find_symbols_by_name(
        self, name: str, *, kind: str | None = None, file_path: str | None = None
    ) -> list[SymbolNode]:
        filters = ["s.name = $name"]
        params: dict[str, Any] = {"name": name}
        if kind:
            filters.append("s.kind = $kind")
            params["kind"] = kind
        if file_path:
            filters.append("s.file_path STARTS WITH $file_path")
            params["file_path"] = file_path
        cypher = f"MATCH (s:Symbol) WHERE {' AND '.join(filters)} RETURN s"
        with self._session() as session:
            result = session.run(cypher, **params)
            return [self._record_to_symbol(r["s"]) for r in result]

    def find_symbols_in_file(self, file_id: str) -> list[SymbolNode]:
        cypher = "MATCH (s:Symbol {file_id: $file_id}) RETURN s"
        with self._session() as session:
            result = session.run(cypher, file_id=file_id)
            return [self._record_to_symbol(r["s"]) for r in result]

    # ------------------------------------------------------------------
    # Traversal
    # ------------------------------------------------------------------

    def get_callers(self, symbol_id: str, depth: int = 1) -> list[SymbolNode]:
        cypher = f"""
            MATCH (caller:Symbol)-[:CALLS*1..{depth}]->(s:Symbol {{id: $id}})
            RETURN DISTINCT caller
        """
        with self._session() as session:
            result = session.run(cypher, id=symbol_id)
            return [self._record_to_symbol(r["caller"]) for r in result]

    def get_callees(self, symbol_id: str, depth: int = 1) -> list[SymbolNode]:
        cypher = f"""
            MATCH (s:Symbol {{id: $id}})-[:CALLS*1..{depth}]->(callee:Symbol)
            RETURN DISTINCT callee
        """
        with self._session() as session:
            result = session.run(cypher, id=symbol_id)
            return [self._record_to_symbol(r["callee"]) for r in result]

    def get_call_path(
        self, from_symbol_id: str, to_symbol_id: str
    ) -> list[SymbolNode] | None:
        cypher = """
            MATCH path = shortestPath(
                (a:Symbol {id: $from_id})-[:CALLS*]->(b:Symbol {id: $to_id})
            )
            RETURN [node in nodes(path) | node] AS symbols
        """
        with self._session() as session:
            result = session.run(cypher, from_id=from_symbol_id, to_id=to_symbol_id)
            record = result.single()
            if record:
                return [self._record_to_symbol(n) for n in record["symbols"]]
        return None

    def get_import_graph(self, file_id: str) -> dict[str, list[str]]:
        cypher = """
            MATCH (f:File {id: $file_id})-[:IMPORTS*0..10]->(dep:File)
            WITH collect(DISTINCT dep) AS deps
            UNWIND deps AS d
            MATCH (d)-[:IMPORTS]->(imp:File)
            RETURN d.path AS source, collect(imp.path) AS targets
        """
        result_map: dict[str, list[str]] = {}
        with self._session() as session:
            result = session.run(cypher, file_id=file_id)
            for record in result:
                result_map[record["source"]] = list(record["targets"])
        return result_map

    def get_subclasses(self, symbol_id: str) -> list[SymbolNode]:
        cypher = """
            MATCH (sub:Symbol)-[:INHERITS_FROM]->(s:Symbol {id: $id})
            RETURN sub
        """
        with self._session() as session:
            result = session.run(cypher, id=symbol_id)
            return [self._record_to_symbol(r["sub"]) for r in result]

    def get_superclasses(self, symbol_id: str) -> list[SymbolNode]:
        cypher = """
            MATCH (s:Symbol {id: $id})-[:INHERITS_FROM]->(sup:Symbol)
            RETURN sup
        """
        with self._session() as session:
            result = session.run(cypher, id=symbol_id)
            return [self._record_to_symbol(r["sup"]) for r in result]

    def get_symbol_context(self, symbol_id: str) -> SymbolContext | None:
        cypher = """
            MATCH (s:Symbol {id: $id})
            OPTIONAL MATCH (f:File)-[:DEFINES]->(s)
            OPTIONAL MATCH (parent:Symbol)-[:CONTAINS]->(s)
            OPTIONAL MATCH (s)-[:CONTAINS]->(method:Symbol)
            OPTIONAL MATCH (caller:Symbol)-[:CALLS]->(s)
            OPTIONAL MATCH (s)-[:CALLS]->(callee:Symbol)
            OPTIONAL MATCH (s)-[:INHERITS_FROM]->(sup:Symbol)
            OPTIONAL MATCH (sub:Symbol)-[:INHERITS_FROM]->(s)
            OPTIONAL MATCH (f)-[:IMPORTS]->(imp_file:File)
            OPTIONAL MATCH (importer:File)-[:IMPORTS]->(f)
            RETURN s, f,
                   collect(DISTINCT parent)[0] AS parent_class,
                   collect(DISTINCT method) AS methods,
                   collect(DISTINCT caller) AS callers,
                   collect(DISTINCT callee) AS callees,
                   collect(DISTINCT sup) AS inherits_from,
                   collect(DISTINCT sub) AS subclasses,
                   collect(DISTINCT imp_file) AS file_imports,
                   collect(DISTINCT importer) AS file_imported_by
        """
        with self._session() as session:
            result = session.run(cypher, id=symbol_id)
            record = result.single()
            if record is None or record["s"] is None:
                return None

            sn = self._record_to_symbol(record["s"])
            fn = self._record_to_file(record["f"]) if record["f"] else None
            if fn is None:
                return None

            return SymbolContext(
                symbol=sn,
                file=fn,
                parent_class=(
                    self._record_to_symbol(record["parent_class"])
                    if record["parent_class"]
                    else None
                ),
                methods=[self._record_to_symbol(m) for m in (record["methods"] or [])],
                callers=[self._record_to_symbol(c) for c in (record["callers"] or [])],
                callees=[self._record_to_symbol(c) for c in (record["callees"] or [])],
                inherits_from=[self._record_to_symbol(s) for s in (record["inherits_from"] or [])],
                subclasses=[self._record_to_symbol(s) for s in (record["subclasses"] or [])],
                file_imports=[self._record_to_file(f) for f in (record["file_imports"] or [])],
                file_imported_by=[
                    self._record_to_file(f) for f in (record["file_imported_by"] or [])
                ],
            )

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_stats(self) -> GraphStats:
        cypher = """
            MATCH (f:File) WITH count(f) AS files
            MATCH (s:Symbol) WITH files, count(s) AS symbols
            MATCH (d:Directory) WITH files, symbols, count(d) AS dirs
            MATCH ()-[r]-() WITH files, symbols, dirs, count(r) AS edges
            RETURN files, symbols, dirs, edges
        """
        with self._session() as session:
            record = session.run(cypher).single()
            total_files = int(record["files"]) if record else 0
            total_symbols = int(record["symbols"]) if record else 0
            total_dirs = int(record["dirs"]) if record else 0
            total_edges = int(record["edges"]) if record else 0

        # Edges by type
        edges_by_type: dict[str, int] = {}
        with self._session() as session:
            result = session.run("MATCH ()-[r]->() RETURN type(r) AS t, count(r) AS c")
            for row in result:
                edges_by_type[row["t"]] = int(row["c"])

        # Symbols by kind
        symbols_by_kind: dict[str, int] = {}
        with self._session() as session:
            result = session.run("MATCH (s:Symbol) RETURN s.kind AS k, count(s) AS c")
            for row in result:
                symbols_by_kind[row["k"]] = int(row["c"])

        # Languages
        languages: dict[str, int] = {}
        with self._session() as session:
            result = session.run("MATCH (f:File) RETURN f.language AS l, count(f) AS c")
            for row in result:
                languages[row["l"]] = int(row["c"])

        return GraphStats(
            total_files=total_files,
            total_symbols=total_symbols,
            total_directories=total_dirs,
            total_edges=total_edges,
            edges_by_type=edges_by_type,
            symbols_by_kind=symbols_by_kind,
            languages=languages,
            graph_store=self.name,
        )

    @property
    def name(self) -> str:
        return "Neo4jGraphStore"

    def close(self) -> None:
        """Close the Neo4j driver connection."""
        self._driver.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _session(self) -> Any:
        return self._driver.session(database=self._database)

    def _batch_run(self, cypher: str, rows: list[dict[str, Any]]) -> None:
        """Run ``cypher`` in batches of ``self._batch_size`` rows."""
        for i in range(0, len(rows), self._batch_size):
            batch = rows[i : i + self._batch_size]
            with self._session() as session:
                session.run(cypher, rows=batch)

    @staticmethod
    def _record_to_symbol(node: Any) -> SymbolNode:
        return SymbolNode(
            id=node["id"],
            name=node["name"],
            qualified_name=node["qualified_name"],
            kind=SymbolKind(node["kind"]),
            file_id=node["file_id"],
            file_path=node["file_path"],
            start_line=node["start_line"],
            end_line=node["end_line"],
            parent_name=node.get("parent_name"),
            docstring=node.get("docstring"),
            decorators=list(node.get("decorators") or []),
        )

    @staticmethod
    def _record_to_file(node: Any) -> FileNode:
        return FileNode(
            id=node["id"],
            path=node["path"],
            language=Language(node["language"]),
            sha256=node["sha256"],
        )


# Avoid unused import warning; GraphEdge is used inside _upsert_edges type comment
from code_indexer.graph.models import GraphEdge as GraphEdge  # noqa: E402, F401
