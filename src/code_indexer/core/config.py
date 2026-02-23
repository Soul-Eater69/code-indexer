"""
Application configuration.

We use Pydantic-Settings so that every setting can be supplied via:
  1. A ``.env`` file (loaded automatically from the working directory).
  2. Real environment variables (these override the .env file).
  3. Direct constructor arguments (useful in tests).

Hierarchy (highest priority first): env-vars > .env > defaults.

Usage::

    from code_indexer.core.config import get_settings
    settings = get_settings()           # cached singleton
    print(settings.embedding.model)
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------------------------
# Sub-models (nested configuration groups)
# ---------------------------------------------------------------------------


class ParserSettings(BaseSettings):
    """Settings for the Tree-sitter parser layer.

    Attributes:
        max_file_size_bytes:
            Files larger than this are skipped during indexing to avoid
            memory exhaustion.  Default 1 MB.
        context_lines:
            Number of lines of surrounding code to capture as
            ``context_before`` / ``context_after`` on each chunk.
            Set to 0 to disable.
        supported_extensions:
            Map of file extension → language string.
            You can extend this at runtime by editing the .env file.
    """

    max_file_size_bytes: int = Field(default=1_048_576, description="Max bytes per file (1 MB)")
    context_lines: int = Field(default=3, ge=0, le=20)
    ignore_patterns: list[str] = Field(
        default=[
            "*.min.js",
            "*.bundle.js",
            "*.map",
            "node_modules/**",
            ".git/**",
            "__pycache__/**",
            "*.pyc",
            "dist/**",
            "build/**",
            ".venv/**",
            "venv/**",
            "*.egg-info/**",
        ],
        description="Glob patterns for files/dirs to skip.",
    )

    model_config = SettingsConfigDict(env_prefix="PARSER_", env_file=".env", extra="ignore")


class ChunkerSettings(BaseSettings):
    """Settings for all chunking strategies.

    Attributes:
        strategy:
            Default chunking strategy when none is specified per request.
        chunk_size:
            Target chunk size in tokens (for token-based strategies) or
            characters (for sliding-window).
        chunk_overlap:
            Number of tokens/characters of overlap between consecutive
            chunks.  Helps preserve context at boundaries.
        min_chunk_size:
            Chunks smaller than this (in tokens) are discarded or merged.
        max_chunk_size:
            Hard upper bound.  Chunks exceeding this are split further.
        ast_node_types:
            Which AST node types the AST chunker should extract.
            Order matters: nodes listed earlier take precedence when
            a node matches multiple types.
    """

    strategy: Literal["ast", "token", "sliding_window", "semantic"] = "ast"
    chunk_size: int = Field(default=512, ge=32, le=8192)
    chunk_overlap: int = Field(default=64, ge=0, le=512)
    min_chunk_size: int = Field(default=20, ge=1)
    max_chunk_size: int = Field(default=2048, ge=64)
    ast_node_types: list[str] = Field(
        default=[
            "function_definition",
            "class_definition",
            "method_definition",
            "function_declaration",
            "class_declaration",
            "interface_declaration",
            "enum_declaration",
            "impl_item",          # Rust impl blocks
            "trait_item",         # Rust traits
            "function_item",      # Rust functions
        ]
    )

    model_config = SettingsConfigDict(env_prefix="CHUNKER_", env_file=".env", extra="ignore")


class EmbeddingSettings(BaseSettings):
    """Settings for the embedding backend.

    Attributes:
        provider:    Which embedding backend to use.
        model:       Model identifier (provider-specific).
        batch_size:  How many chunks to embed in a single API/model call.
                     Larger batches are more efficient but use more memory.
        dimensions:  Expected output dimension.  Used for validation and
                     to pre-allocate buffers.
        openai_api_key:  OpenAI API key (only needed for openai provider).
        ollama_base_url: Base URL of a running Ollama server.
    """

    provider: Literal["openai", "sentence_transformer", "ollama"] = "sentence_transformer"
    model: str = "all-MiniLM-L6-v2"
    batch_size: int = Field(default=64, ge=1, le=512)
    dimensions: int = Field(default=384, ge=64)
    openai_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"

    model_config = SettingsConfigDict(env_prefix="EMBEDDING_", env_file=".env", extra="ignore")


class VectorStoreSettings(BaseSettings):
    """Settings for the vector store backend.

    Attributes:
        provider:          Which vector store to use.
        collection_name:   Name of the collection / namespace inside the store.
        chroma_persist_dir: Directory for ChromaDB's on-disk persistence.
        qdrant_url:        URL for a Qdrant server (HTTP).
        qdrant_api_key:    Optional Qdrant API key for cloud deployments.
        distance_metric:   Vector distance function.
    """

    provider: Literal["chroma", "qdrant", "in_memory"] = "chroma"
    collection_name: str = "code_index"
    chroma_persist_dir: str = ".chroma"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    distance_metric: Literal["cosine", "dot", "euclidean"] = "cosine"

    model_config = SettingsConfigDict(env_prefix="VECTORSTORE_", env_file=".env", extra="ignore")


class GraphSettings(BaseSettings):
    """Settings for the code knowledge graph layer.

    Attributes:
        enabled:       Whether to build the graph index alongside the vector index.
        provider:      Graph store backend.
        neo4j_uri:     Bolt URI for Neo4j (only used when provider="neo4j").
        neo4j_username: Neo4j username.
        neo4j_password: Neo4j password.
        neo4j_database: Neo4j database name.
    """

    enabled: bool = False
    provider: Literal["in_memory", "neo4j"] = "in_memory"
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: str = "password"
    neo4j_database: str = "neo4j"

    model_config = SettingsConfigDict(env_prefix="GRAPH_", env_file=".env", extra="ignore")


class LLMSettings(BaseSettings):
    """Settings for the LLM client used by TermEnricher and ConcernClusterer.

    Attributes:
        provider:       Which LLM backend to use.
        model:          Model for cheap bulk operations (expansion, definition,
                        summarisation, ranking).  Default: ``gpt-4o-mini``.
        cluster_model:  Stronger model for concern clustering (one call per
                        query).  Default: same as ``model``.
        api_key:        API key (OpenAI) or ignored (Ollama).
        base_url:       Base URL for Ollama server.
        max_tokens:     Default max output tokens per LLM call.
        temperature:    Sampling temperature (0 = deterministic).
        concurrency:    Number of LLM calls to run in parallel during offline
                        enrichment.  Higher values reduce wall-clock time but
                        increase API rate-limit pressure.  Default: ``20``.
    """

    provider: Literal["openai", "ollama"] = "openai"
    model: str = "gpt-4o-mini"
    cluster_model: str = ""          # empty → fall back to ``model``
    api_key: str = ""
    base_url: str = "http://localhost:11434"
    max_tokens: int = Field(default=512, ge=16, le=4096)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    concurrency: int = Field(default=20, ge=1, le=200)

    model_config = SettingsConfigDict(env_prefix="LLM_", env_file=".env", extra="ignore")


class APISettings(BaseSettings):
    """Settings for the FastAPI server.

    Attributes:
        host:    Bind address for uvicorn.
        port:    TCP port.
        workers: Number of uvicorn worker processes.
        reload:  Enable hot-reload (dev only; must be False in production).
        api_key: Optional bearer token.  When non-empty, all API requests
                 must include ``Authorization: Bearer <api_key>``.
        cors_origins: List of allowed CORS origins.
    """

    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535)
    workers: int = Field(default=1, ge=1, le=16)
    reload: bool = False
    api_key: str = ""
    cors_origins: list[str] = Field(default=["*"])
    log_level: Literal["debug", "info", "warning", "error"] = "info"

    model_config = SettingsConfigDict(env_prefix="API_", env_file=".env", extra="ignore")

    @field_validator("reload")
    @classmethod
    def _reload_only_in_dev(cls, v: bool) -> bool:
        # This is a soft guard; callers can still override.
        return v


# ---------------------------------------------------------------------------
# Root settings object
# ---------------------------------------------------------------------------


class Settings(BaseSettings):
    """Root application settings.

    All sub-groups are instantiated with their own env-prefix, so you can
    configure each layer independently::

        PARSER_MAX_FILE_SIZE_BYTES=2097152
        CHUNKER_STRATEGY=token
        EMBEDDING_PROVIDER=openai
        EMBEDDING_OPENAI_API_KEY=sk-...
        VECTORSTORE_PROVIDER=qdrant
        API_PORT=9000
    """

    parser: ParserSettings = Field(default_factory=ParserSettings)
    chunker: ChunkerSettings = Field(default_factory=ChunkerSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    vectorstore: VectorStoreSettings = Field(default_factory=VectorStoreSettings)
    graph: GraphSettings = Field(default_factory=GraphSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    api: APISettings = Field(default_factory=APISettings)

    # Global toggles
    debug: bool = False
    log_format: Literal["json", "console"] = "console"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached application settings singleton.

    The cache is keyed by Python process lifetime.  In tests, call
    ``get_settings.cache_clear()`` before constructing a fresh instance.
    """
    return Settings()
