"""
Tree-sitter based AST parser.

How Tree-sitter works (brief primer)
-------------------------------------
Tree-sitter is an incremental parser generator.  For each language there is
a compiled C shared library (grammar) that can parse source code into a
concrete syntax tree (CST).  The Python bindings expose this through the
``tree_sitter`` package.

Parsing pipeline inside this module:

  raw bytes
      │
      ▼
  tree_sitter.Parser.parse(bytes, language)
      │  Returns a ``tree_sitter.Tree``
      ▼
  _walk_tree(tree.root_node)
      │  Recursively visits every node
      │  Filters to "interesting" node types (function_definition, etc.)
      ▼
  list[ASTNode]   (our serialisable model, not the C binding object)

The filtering is intentionally permissive at this stage: we extract
*everything* that might be useful and let the chunker decide what to keep.

Error tolerance
---------------
Tree-sitter is error-tolerant: it always produces a tree even for syntactically
invalid code.  Error nodes are of type ``"ERROR"`` or ``"MISSING"``.  We collect
them as non-fatal parse errors and still return whatever valid nodes were found.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from code_indexer.core.models import ASTNode, Language, ParsedFile, SourceFile
from code_indexer.parsers.base import BaseParser
from code_indexer.parsers.language_registry import get_tree_sitter_language

if TYPE_CHECKING:
    pass  # kept for future type-only imports

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Node types we want to extract from each language's AST.
# Tree-sitter grammar node type names are language-specific but follow
# predictable conventions.  We use a superset here; unknown types are ignored.
# ---------------------------------------------------------------------------

_INTERESTING_NODE_TYPES: frozenset[str] = frozenset(
    {
        # Python
        "function_definition",
        "async_function_def",
        "class_definition",
        "decorated_definition",  # wraps functions/classes with decorators
        # JavaScript / TypeScript
        "function_declaration",
        "function_expression",
        "arrow_function",
        "class_declaration",
        "class_expression",
        "method_definition",
        "interface_declaration",  # TypeScript
        "type_alias_declaration",  # TypeScript
        "enum_declaration",        # TypeScript
        "abstract_class_declaration",
        # Rust
        "function_item",
        "impl_item",
        "trait_item",
        "struct_item",
        "enum_item",
        "mod_item",
        # Go
        "function_declaration",
        "method_declaration",
        "type_declaration",
        "type_spec",
        # Java
        "method_declaration",
        "constructor_declaration",
        "class_declaration",
        "interface_declaration",
        "enum_declaration",
        # C / C++
        "function_definition",
        "declaration",
        "struct_specifier",
        "class_specifier",
    }
)

# AST node types that carry the identifier name of the construct.
# We look inside these child node types to find the name.
_NAME_CHILD_TYPES: frozenset[str] = frozenset(
    {"identifier", "type_identifier", "property_identifier", "field_identifier"}
)


class TreeSitterParser(BaseParser):
    """Parse source files into AST node trees using Tree-sitter.

    One ``TreeSitterParser`` instance can be reused across many files and is
    thread-safe (Tree-sitter parsers are not, so we create a new
    ``tree_sitter.Parser`` per ``parse`` call to avoid locking overhead).

    Args:
        max_depth:
            Maximum recursion depth when walking the AST.  Limiting this
            prevents stack overflows on pathologically deeply nested code.
        extract_types:
            Override the default set of interesting node types.
    """

    def __init__(
        self,
        max_depth: int = 20,
        extract_types: frozenset[str] | None = None,
    ) -> None:
        self._max_depth = max_depth
        self._extract_types = extract_types or _INTERESTING_NODE_TYPES

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(self, source_file: SourceFile) -> ParsedFile:
        """Parse *source_file* using Tree-sitter and return a ``ParsedFile``.

        If the language has no registered grammar (e.g. ``Language.UNKNOWN``),
        we return a ``ParsedFile`` with an empty node list and a warning.

        Args:
            source_file: The file to parse.

        Returns:
            ``ParsedFile`` with extracted AST nodes and any parse errors.
        """
        import tree_sitter  # imported here so the module can load without it

        ts_language = get_tree_sitter_language(source_file.language)
        if ts_language is None:
            logger.debug(
                "No Tree-sitter grammar for %s (%s), skipping AST parse",
                source_file.path,
                source_file.language.value,
            )
            return ParsedFile(
                source=source_file,
                ast_nodes=[],
                parse_errors=[
                    f"No grammar available for language: {source_file.language.value}"
                ],
            )

        # Create a fresh parser for this call (thread-safety).
        parser = tree_sitter.Parser(ts_language)
        raw_bytes = source_file.content.encode("utf-8", errors="replace")
        tree = parser.parse(raw_bytes)

        # Collect parse errors (ERROR nodes in the CST).
        parse_errors: list[str] = []
        self._collect_errors(tree.root_node, parse_errors)

        # Walk the CST and extract interesting nodes.
        ast_nodes: list[ASTNode] = []
        self._walk_tree(
            node=tree.root_node,
            source_bytes=raw_bytes,
            results=ast_nodes,
            depth=0,
        )

        logger.debug(
            "Parsed %s: %d AST nodes, %d errors",
            source_file.path,
            len(ast_nodes),
            len(parse_errors),
        )

        return ParsedFile(
            source=source_file,
            ast_nodes=ast_nodes,
            parse_errors=parse_errors,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _walk_tree(
        self,
        node: object,
        source_bytes: bytes,
        results: list[ASTNode],
        depth: int,
    ) -> None:
        """Recursively walk the Tree-sitter CST, collecting interesting nodes.

        We use an iterative DFS with an explicit stack to avoid Python's
        default recursion limit.

        Args:
            node:         The current Tree-sitter ``Node`` (C binding object).
            source_bytes: Raw UTF-8 bytes of the source file (needed to
                          extract text from byte offsets).
            results:      Accumulator list for extracted ``ASTNode`` objects.
            depth:        Current depth in the AST.
        """
        # Iterative DFS stack: (ts_node, current_depth)
        stack: list[tuple[object, int]] = [(node, depth)]

        while stack:
            current_node, current_depth = stack.pop()

            if current_depth > self._max_depth:
                continue

            if current_node.type in self._extract_types:  # type: ignore[union-attr]
                ast_node = self._convert_node(current_node, source_bytes, current_depth)
                results.append(ast_node)
                # We still recurse into children so nested classes/functions
                # inside a class body are also captured.

            # Push children in reverse order so they are processed left-to-right.
            for child in reversed(current_node.children):  # type: ignore[union-attr]
                stack.append((child, current_depth + 1))

    def _convert_node(
        self,
        node: object,
        source_bytes: bytes,
        depth: int,
    ) -> ASTNode:
        """Convert a Tree-sitter ``Node`` into our serialisable ``ASTNode``.

        Args:
            node:         Tree-sitter Node.
            source_bytes: Raw file bytes for text extraction.
            depth:        Depth in the AST.

        Returns:
            An ``ASTNode`` populated with offsets, name, and type.
        """
        # node.start_point / end_point are (row, col) tuples, 0-indexed.
        start_line: int = node.start_point[0] + 1  # type: ignore[index]
        end_line: int = node.end_point[0] + 1      # type: ignore[index]
        start_byte: int = node.start_byte            # type: ignore[attr-defined]
        end_byte: int = node.end_byte                # type: ignore[attr-defined]

        name = self._extract_name(node, source_bytes)

        return ASTNode(
            type=node.type,                          # type: ignore[union-attr]
            name=name,
            start_byte=start_byte,
            end_byte=end_byte,
            start_line=start_line,
            end_line=end_line,
            depth=depth,
        )

    def _extract_name(self, node: object, source_bytes: bytes) -> str:
        """Extract the identifier name from an AST node.

        For a ``function_definition`` node, this returns the function name.
        For a ``class_definition``, the class name.  etc.

        We look at direct children of the node for an ``identifier`` child
        whose text gives us the name.

        Args:
            node:         Tree-sitter Node.
            source_bytes: Raw file bytes.

        Returns:
            The name string, or empty string if not determinable.
        """
        for child in node.children:  # type: ignore[union-attr]
            if child.type in _NAME_CHILD_TYPES:
                try:
                    return source_bytes[child.start_byte : child.end_byte].decode(
                        "utf-8", errors="replace"
                    )
                except Exception:  # noqa: BLE001
                    return ""
        return ""

    def _collect_errors(self, node: object, errors: list[str]) -> None:
        """Recursively collect ERROR nodes from the CST into *errors*.

        Tree-sitter marks syntactically invalid sections as ``ERROR`` or
        ``MISSING`` nodes.  We record their location but do not raise.

        Args:
            node:   Tree-sitter Node.
            errors: Accumulator list.
        """
        if node.type in ("ERROR", "MISSING"):  # type: ignore[union-attr]
            errors.append(
                f"Parse error at line {node.start_point[0] + 1}:"  # type: ignore[index]
                f"{node.start_point[1]} (node type: {node.type})"  # type: ignore[index, union-attr]
            )
        for child in node.children:  # type: ignore[union-attr]
            self._collect_errors(child, errors)
