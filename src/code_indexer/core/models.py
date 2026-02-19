"""
Core data models for the code indexing system.

All data flowing through the pipeline is represented as Pydantic models,
giving us runtime validation, serialisation to/from JSON, and clear contracts
between pipeline stages.

Data-flow summary:
  SourceFile -> ParsedFile -> [Chunk] -> [EmbeddedChunk] -> VectorStore
                                                            ↓
                                                       SearchResult
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class Language(str, Enum):
    """Supported programming languages.

    Each value is the canonical identifier used internally and by Tree-sitter
    grammar packages (e.g. ``tree_sitter_python``).
    """

    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    RUST = "rust"
    GO = "go"
    JAVA = "java"
    CPP = "cpp"
    C = "c"
    UNKNOWN = "unknown"


class ChunkType(str, Enum):
    """Semantic type of a code chunk.

    AST-aware chunkers assign fine-grained types; token/sliding-window
    chunkers fall back to ``SNIPPET``.
    """

    # Structural units emitted by AST-aware chunkers
    MODULE = "module"         # Top-level file scope
    CLASS = "class"           # Class / struct definition
    FUNCTION = "function"     # Function / method definition
    METHOD = "method"         # Method inside a class (sub-type of FUNCTION)
    COMMENT = "comment"       # Doc-comment block
    IMPORT = "import"         # Import / use / include block
    INTERFACE = "interface"   # Interface / trait / protocol
    ENUM = "enum"             # Enum definition
    CONSTANT = "constant"     # Module-level constant

    # Generic units emitted by token / sliding-window chunkers
    SNIPPET = "snippet"


class ChunkerStrategy(str, Enum):
    """Available chunking strategies.

    These correspond to concrete ``BaseChunker`` subclasses.
    """

    AST = "ast"                         # Tree-sitter AST-aware
    TOKEN = "token"                     # Token-count-aware sliding window
    SLIDING_WINDOW = "sliding_window"   # Character sliding window
    SEMANTIC = "semantic"               # Sentence-transformer semantic boundary


# ---------------------------------------------------------------------------
# Source file model
# ---------------------------------------------------------------------------


class SourceFile(BaseModel):
    """Represents a single source file to be indexed.

    Attributes:
        id:          Stable deterministic ID derived from the file path.
        path:        Absolute or repo-relative path of the file.
        content:     Raw text content of the file.
        language:    Detected programming language.
        encoding:    Character encoding detected by chardet.
        size_bytes:  File size in bytes.
        sha256:      SHA-256 hex digest of the raw bytes (for dedup).
        repo_root:   Optional absolute path of the repository root.
                     When set, ``path`` is stored relative to this root.
        metadata:    Arbitrary extra key-value pairs (e.g. git author, branch).
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    path: str
    content: str
    language: Language = Language.UNKNOWN
    encoding: str = "utf-8"
    size_bytes: int = 0
    sha256: str = ""
    repo_root: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    indexed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def _compute_derived_fields(self) -> SourceFile:
        raw = self.content.encode(self.encoding, errors="replace")
        if not self.size_bytes:
            self.size_bytes = len(raw)
        if not self.sha256:
            self.sha256 = hashlib.sha256(raw).hexdigest()
        return self


# ---------------------------------------------------------------------------
# AST node model
# ---------------------------------------------------------------------------


class ASTNode(BaseModel):
    """Lightweight representation of a Tree-sitter AST node.

    We don't store the full Tree-sitter node object (not serialisable), but
    extract the fields we need for downstream chunking and metadata tagging.

    Attributes:
        type:         Tree-sitter node type (e.g. ``function_definition``).
        name:         Human-readable identifier extracted from the node
                      (e.g. function / class name). Empty string if N/A.
        start_byte:   Byte offset of the first character in the source file.
        end_byte:     Byte offset *past* the last character.
        start_line:   1-indexed line number where the node begins.
        end_line:     1-indexed line number where the node ends.
        children:     Nested child nodes (for hierarchical representations).
        depth:        Depth in the AST from the root (0 = file root).
    """

    type: str
    name: str = ""
    start_byte: int
    end_byte: int
    start_line: int
    end_line: int
    children: list[ASTNode] = Field(default_factory=list)
    depth: int = 0

    @property
    def byte_length(self) -> int:
        return self.end_byte - self.start_byte

    @property
    def line_count(self) -> int:
        return self.end_line - self.start_line + 1


# ---------------------------------------------------------------------------
# Parsed file model
# ---------------------------------------------------------------------------


class ParsedFile(BaseModel):
    """The result of running a parser over a ``SourceFile``.

    Attributes:
        source:       The original source file.
        ast_nodes:    Top-level AST nodes extracted from the file.
        parse_errors: Any Tree-sitter parse error messages encountered.
                      A non-empty list means partial parsing (Tree-sitter is
                      error-tolerant and always returns a tree, even for
                      syntactically invalid code).
    """

    source: SourceFile
    ast_nodes: list[ASTNode] = Field(default_factory=list)
    parse_errors: list[str] = Field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return len(self.parse_errors) > 0


# ---------------------------------------------------------------------------
# Chunk model
# ---------------------------------------------------------------------------


class Chunk(BaseModel):
    """A single indexable unit of code extracted from a ``SourceFile``.

    A chunk carries both the raw text and all the metadata required to:
    * embed it into a vector space (text),
    * retrieve it and render it to users (text + metadata),
    * and link it back to its origin for attribution (source_file_id, path).

    Attributes:
        id:             Stable deterministic ID.  Derived from the source
                        file ID and byte offsets so re-indexing the same
                        content always yields the same chunk ID.
        source_file_id: ID of the parent ``SourceFile``.
        path:           File path (duplicated here for query convenience).
        language:       Inherited from the source file.
        chunk_type:     Semantic type (function, class, snippet, …).
        name:           Human-readable name (e.g. class/function name).
                        Empty string for generic snippets.
        content:        The raw text of this chunk.
        token_count:    Approximate token count using the configured tokenizer.
        start_line:     1-indexed start line inside the source file.
        end_line:       1-indexed end line inside the source file.
        start_byte:     Byte offset of the first character.
        end_byte:       Byte offset past the last character.
        context_before: Optional lines of code preceding this chunk.
                        Useful for prompts that need surrounding context.
        context_after:  Optional lines of code following this chunk.
        metadata:       Arbitrary extra key-value pairs inherited from the
                        source file plus any chunker-specific additions.
    """

    id: str = ""
    source_file_id: str
    path: str
    language: Language
    chunk_type: ChunkType = ChunkType.SNIPPET
    name: str = ""
    content: str
    token_count: int = 0
    start_line: int = 0
    end_line: int = 0
    start_byte: int = 0
    end_byte: int = 0
    context_before: str = ""
    context_after: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def _set_id(self) -> Chunk:
        if not self.id:
            # Deterministic: same file + same byte range → same chunk ID.
            key = f"{self.source_file_id}:{self.start_byte}:{self.end_byte}"
            self.id = hashlib.sha256(key.encode()).hexdigest()[:32]
        return self

    @field_validator("content")
    @classmethod
    def _content_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Chunk content must not be blank")
        return v

    def to_embedding_text(self, *, include_path: bool = True) -> str:
        """Build the text string that will be sent to the embedding model.

        We prepend a structured header so the model sees contextual signals:

            [python] [function] src/auth/utils.py
            def verify_token(token: str) -> bool:
                ...
        """
        parts: list[str] = []
        if include_path:
            header = f"[{self.language.value}] [{self.chunk_type.value}] {self.path}"
            if self.name:
                header += f" :: {self.name}"
            parts.append(header)
        parts.append(self.content)
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Embedded chunk model
# ---------------------------------------------------------------------------


class EmbeddedChunk(BaseModel):
    """A ``Chunk`` paired with its vector embedding.

    Attributes:
        chunk:      The original chunk.
        embedding:  Dense vector produced by the embedding backend.
                    Dimensionality depends on the model (e.g. 1536 for
                    text-embedding-3-small, 384 for all-MiniLM-L6-v2).
        model:      Identifier of the embedding model used.
    """

    chunk: Chunk
    embedding: list[float]
    model: str

    @property
    def dimension(self) -> int:
        return len(self.embedding)


# ---------------------------------------------------------------------------
# Search / retrieval models
# ---------------------------------------------------------------------------


class SearchQuery(BaseModel):
    """Parameters for a vector similarity search.

    Attributes:
        query:          Natural language or code query string.
        top_k:          Maximum number of results to return.
        language:       Optional filter: only return chunks in this language.
        chunk_types:    Optional allow-list of chunk types.
        path_prefix:    Optional path prefix filter (e.g. ``src/auth/``).
        min_score:      Minimum cosine similarity threshold [0, 1].
        include_context: Whether to include context_before / context_after
                         in returned chunks.
        metadata_filters: Arbitrary metadata key-value equality filters.
    """

    query: str
    top_k: int = Field(default=10, ge=1, le=100)
    language: Language | None = None
    chunk_types: list[ChunkType] | None = None
    path_prefix: str | None = None
    min_score: float = Field(default=0.0, ge=0.0, le=1.0)
    include_context: bool = False
    metadata_filters: dict[str, Any] = Field(default_factory=dict)


class SearchResult(BaseModel):
    """A single result returned from a vector similarity search.

    Attributes:
        chunk:   The matching chunk.
        score:   Cosine similarity score in [0, 1].  Higher is more similar.
        rank:    1-indexed position in the result list (set after reranking).
    """

    chunk: Chunk
    score: float
    rank: int = 0


class SearchResponse(BaseModel):
    """Full response from a search request.

    Attributes:
        query:    The original query string.
        results:  Ordered list of search results (best match first).
        total:    Total number of chunks searched.
        latency_ms: Wall-clock time for the search, in milliseconds.
    """

    query: str
    results: list[SearchResult]
    total: int = 0
    latency_ms: float = 0.0


# ---------------------------------------------------------------------------
# Index statistics
# ---------------------------------------------------------------------------


class IndexStats(BaseModel):
    """Summary statistics for an index.

    Returned by the ``/index/stats`` API endpoint and the ``stats`` CLI
    command.
    """

    total_files: int = 0
    total_chunks: int = 0
    total_tokens: int = 0
    languages: dict[str, int] = Field(default_factory=dict)   # language → chunk count
    chunk_types: dict[str, int] = Field(default_factory=dict) # type → chunk count
    embedding_model: str = ""
    vector_store: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
