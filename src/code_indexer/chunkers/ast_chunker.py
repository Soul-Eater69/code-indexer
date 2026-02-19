"""
AST-aware chunker: produces one ``Chunk`` per meaningful AST node.

Why AST-aware chunking is superior for code
--------------------------------------------
Token-based chunking splits code at arbitrary token boundaries.  This means
a function definition might be split across two chunks: the signature in one
and the body in another.  The RAG retriever then has to guess whether to
retrieve both chunks to answer a question about the function.

AST-aware chunking respects *semantic boundaries*: each chunk corresponds to
a single function, class, method, etc.  This gives the retriever much cleaner
signal and avoids broken context.

Algorithm
----------
1. Iterate over ``parsed_file.ast_nodes`` (already filtered to interesting
   node types by the parser).
2. For each node, extract the raw source text using byte offsets.
3. If the node text exceeds ``max_chunk_size`` tokens, split it recursively
   using a token chunker as fallback.
4. If the node text is below ``min_chunk_size`` tokens, attach it to the
   next/previous chunk or discard.
5. Add surrounding context lines for each chunk.

``chunk_type`` and ``name`` are inferred from the AST node's type string.
"""

from __future__ import annotations

import logging

from code_indexer.chunkers.base import BaseChunker
from code_indexer.chunkers.token_chunker import TokenChunker
from code_indexer.core.models import Chunk, ChunkType, ParsedFile

logger = logging.getLogger(__name__)

# Mapping from Tree-sitter node type strings → our ``ChunkType`` enum.
# Node types not listed here fall back to ``ChunkType.SNIPPET``.
_NODE_TYPE_MAP: dict[str, ChunkType] = {
    # Python
    "function_definition": ChunkType.FUNCTION,
    "async_function_def": ChunkType.FUNCTION,
    "class_definition": ChunkType.CLASS,
    "decorated_definition": ChunkType.FUNCTION,  # may be class too, fine for now
    # JS/TS
    "function_declaration": ChunkType.FUNCTION,
    "function_expression": ChunkType.FUNCTION,
    "arrow_function": ChunkType.FUNCTION,
    "class_declaration": ChunkType.CLASS,
    "class_expression": ChunkType.CLASS,
    "method_definition": ChunkType.METHOD,
    "interface_declaration": ChunkType.INTERFACE,
    "type_alias_declaration": ChunkType.CLASS,
    "enum_declaration": ChunkType.ENUM,
    "abstract_class_declaration": ChunkType.CLASS,
    # Rust
    "function_item": ChunkType.FUNCTION,
    "impl_item": ChunkType.CLASS,
    "trait_item": ChunkType.INTERFACE,
    "struct_item": ChunkType.CLASS,
    "enum_item": ChunkType.ENUM,
    "mod_item": ChunkType.MODULE,
    # Go
    "function_declaration": ChunkType.FUNCTION,
    "method_declaration": ChunkType.METHOD,
    "type_declaration": ChunkType.CLASS,
    # Java
    "method_declaration": ChunkType.METHOD,
    "constructor_declaration": ChunkType.METHOD,
    "class_declaration": ChunkType.CLASS,
    "interface_declaration": ChunkType.INTERFACE,
    # C/C++
    "function_definition": ChunkType.FUNCTION,
    "struct_specifier": ChunkType.CLASS,
    "class_specifier": ChunkType.CLASS,
}


class ASTChunker(BaseChunker):
    """Produce one ``Chunk`` per semantic AST node (function, class, …).

    Args:
        chunk_size:     Token limit for the AST node text.  Nodes exceeding
                        this are split with ``TokenChunker`` as a fallback.
        chunk_overlap:  Overlap tokens when falling back to ``TokenChunker``.
        min_chunk_size: Nodes below this word count are discarded.
        max_chunk_size: Hard upper bound; triggers ``TokenChunker`` fallback.
        context_lines:  Lines of surrounding code to capture.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        min_chunk_size: int = 20,
        max_chunk_size: int = 2048,
        context_lines: int = 3,
    ) -> None:
        super().__init__(chunk_size, chunk_overlap, min_chunk_size, max_chunk_size)
        self.context_lines = context_lines
        # Fallback chunker for oversized nodes.
        self._fallback = TokenChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_size=min_chunk_size,
            max_chunk_size=max_chunk_size,
        )

    def chunk(self, parsed_file: ParsedFile) -> list[Chunk]:
        """Chunk *parsed_file* by AST node boundaries.

        Args:
            parsed_file: Parsed source file with AST nodes.

        Returns:
            List of ``Chunk`` objects, one per AST node (or sub-chunk for
            oversized nodes).
        """
        if not parsed_file.ast_nodes:
            logger.debug(
                "%s has no AST nodes; falling back to token chunker",
                parsed_file.source.path,
            )
            return self._fallback.chunk(parsed_file)

        source = parsed_file.source
        raw_bytes = source.content.encode("utf-8", errors="replace")
        all_lines = source.content.splitlines()
        chunks: list[Chunk] = []

        for node in parsed_file.ast_nodes:
            # Extract raw text for this node using byte offsets.
            node_text = raw_bytes[node.start_byte : node.end_byte].decode(
                "utf-8", errors="replace"
            )

            if not node_text.strip():
                continue

            chunk_type = _NODE_TYPE_MAP.get(node.type, ChunkType.SNIPPET)

            maybe_chunk = self._build_chunk(
                parsed_file=parsed_file,
                content=node_text,
                start_line=node.start_line,
                end_line=node.end_line,
                start_byte=node.start_byte,
                end_byte=node.end_byte,
                chunk_type=chunk_type,
                name=node.name,
            )

            if maybe_chunk is None:
                logger.debug(
                    "Discarding tiny AST node %r (%s) in %s",
                    node.name,
                    node.type,
                    source.path,
                )
                continue

            # Annotate with surrounding context lines.
            self._add_context(maybe_chunk, all_lines, self.context_lines)
            chunks.append(maybe_chunk)

        logger.debug(
            "ASTChunker: %s → %d chunks",
            source.path,
            len(chunks),
        )
        return chunks
