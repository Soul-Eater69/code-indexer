"""Code knowledge graph layer.

This package implements the structural/relational layer on top of the vector
index.  Where the vector index answers *"what code is semantically similar to
my query?"*, the graph layer answers:

- What does ``function X`` call?
- What calls ``function X``?
- What does ``file A`` import?
- What classes inherit from ``Foo``?
- What is the call chain between ``A`` and ``B``?
- What symbols does ``module M`` export?

The graph is extracted by traversing Tree-sitter CSTs for every indexed file
and recording all structural relationships.  Two storage backends are provided:

* :class:`~code_indexer.graph.in_memory_graph.InMemoryGraphStore` — NetworkX
  MultiDiGraph, zero infrastructure, suitable for development and small repos.
* :class:`~code_indexer.graph.neo4j_store.Neo4jGraphStore` — production-grade
  persistent graph database with full Cypher traversal support.
"""

from code_indexer.graph.models import (
    DirectoryNode,
    FileNode,
    GraphEdge,
    GraphSnapshot,
    GraphStats,
    NodeType,
    RelType,
    SymbolContext,
    SymbolKind,
    SymbolNode,
)

__all__ = [
    "FileNode",
    "SymbolNode",
    "DirectoryNode",
    "GraphEdge",
    "GraphSnapshot",
    "GraphStats",
    "SymbolContext",
    "NodeType",
    "SymbolKind",
    "RelType",
]
