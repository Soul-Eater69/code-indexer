# Code Indexer

> **Production-ready codebase indexing system for RAG and code generation.**
>
> Parse your entire codebase with Tree-sitter, chunk it semantically, embed it
> with the model of your choice, and query it in natural language — all from a
> clean Python library, REST API, or CLI.

---

## Table of Contents

1. [Why Code Indexer?](#why-code-indexer)
2. [Architecture Overview](#architecture-overview)
3. [Pipeline Deep Dive](#pipeline-deep-dive)
   - [Stage 1 — File Walking](#stage-1--file-walking)
   - [Stage 2 — Parsing with Tree-sitter](#stage-2--parsing-with-tree-sitter)
   - [Stage 3 — Chunking](#stage-3--chunking)
   - [Stage 4 — Embedding](#stage-4--embedding)
   - [Stage 5 — Vector Store](#stage-5--vector-store)
   - [Stage 6 — Retrieval](#stage-6--retrieval)
4. [Data Flow Diagram](#data-flow-diagram)
5. [Module Map](#module-map)
6. [Quick Start](#quick-start)
7. [Configuration Reference](#configuration-reference)
8. [API Reference](#api-reference)
9. [CLI Reference](#cli-reference)
10. [Chunking Strategies Explained](#chunking-strategies-explained)
11. [Embedding Backend Comparison](#embedding-backend-comparison)
12. [Vector Store Backend Comparison](#vector-store-backend-comparison)
13. [RAG Integration Guide](#rag-integration-guide)
14. [Deployment Guide](#deployment-guide)
15. [Development Guide](#development-guide)

---

## Why Code Indexer?

| Problem | Code Indexer Solution |
|---|---|
| LLMs forget code context > 200k tokens | Semantic search retrieves only the relevant chunks |
| Naive line-splitting breaks functions in half | AST-aware chunking respects function/class boundaries |
| Different codebases use different languages | Tree-sitter grammars for 8+ languages out of the box |
| Expensive GPU embedding runs | Local `sentence-transformers` or remote OpenAI/Ollama |
| Locked into one vector DB | Pluggable: ChromaDB, Qdrant, or in-memory |
| Complex setup required | One command: `code-indexer index ./myrepo` |

---

## Architecture Overview

The system is built as a **layered pipeline** where each layer has a clean
abstract interface and multiple interchangeable implementations.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          CODE INDEXER SYSTEM                            │
│                                                                         │
│   ┌────────────┐    ┌────────────┐    ┌────────────┐    ┌───────────┐  │
│   │  CLI / API │    │  Indexing  │    │  Retrieval │    │  Vector   │  │
│   │  Layer     │───▶│  Pipeline  │───▶│  Layer     │◀──▶│  Store    │  │
│   └────────────┘    └─────┬──────┘    └────────────┘    └───────────┘  │
│                           │                                             │
│               ┌───────────┼───────────────┐                            │
│               ▼           ▼               ▼                            │
│         ┌──────────┐ ┌──────────┐ ┌──────────────┐                    │
│         │  Parser  │ │ Chunker  │ │  Embedder    │                    │
│         │ (TS AST) │ │ (3 strat)│ │ (3 backends) │                    │
│         └──────────┘ └──────────┘ └──────────────┘                    │
└─────────────────────────────────────────────────────────────────────────┘
```

### Layer Responsibilities

| Layer | What it does | Key files |
|---|---|---|
| **Core** | Pydantic models, config, exceptions | `core/models.py`, `core/config.py` |
| **Parser** | Tree-sitter AST extraction | `parsers/tree_sitter_parser.py` |
| **Chunker** | Split code into indexable chunks | `chunkers/ast_chunker.py`, `token_chunker.py` |
| **Embedder** | Convert text to vectors | `embeddings/openai_embedder.py`, etc. |
| **Vector Store** | Store and retrieve vectors | `vectorstore/chroma_store.py`, etc. |
| **Retriever** | Query the index | `retrieval/retriever.py` |
| **API** | FastAPI REST endpoints | `api/app.py`, `api/routes/` |
| **CLI** | `click`-based terminal interface | `cli/main.py` |

---

## Pipeline Deep Dive

### Stage 1 — File Walking

```
Filesystem
    │
    ▼
CodebaseWalker.walk()
    │   • Recursively scans the root directory
    │   • Applies glob ignore patterns (.gitignore-style)
    │   • Skips binary files (null-byte heuristic)
    │   • Skips files > max_file_size (default 1 MB)
    │   • Detects encoding with chardet
    │   • Detects language from file extension
    ▼
Iterator[SourceFile]
```

**`SourceFile` model** (simplified):
```python
class SourceFile(BaseModel):
    id: str           # UUID, stable per path
    path: str         # repo-relative path
    content: str      # decoded text
    language: Language
    encoding: str     # e.g. "utf-8", "latin-1"
    size_bytes: int
    sha256: str       # for deduplication
    metadata: dict    # arbitrary extra fields
```

The `sha256` field lets you skip re-indexing unchanged files:
if a file's hash matches what's already in the index, you can skip it entirely.

---

### Stage 2 — Parsing with Tree-sitter

#### What is Tree-sitter?

Tree-sitter is an **incremental parser generator**.  For each language, a
C shared library (grammar) is compiled from a formal grammar description.
The Python bindings (`tree-sitter` package) call into these libraries via
ctypes/cffi, making parsing extremely fast (typically < 1ms per file).

#### How it works here

```
SourceFile.content (UTF-8 bytes)
    │
    ▼
tree_sitter.Parser.parse(bytes)         ← one parser per call (thread-safe)
    │
    ▼
tree_sitter.Tree  (concrete syntax tree)
    │
    ├── root_node
    │       ├── function_definition [0:42]
    │       │       ├── identifier "add"
    │       │       ├── parameters
    │       │       └── block ...
    │       └── class_definition [44:120]
    │               ├── identifier "Calculator"
    │               └── block ...
    │
    ▼
_walk_tree()     ← iterative DFS (no recursion limit)
    │   Filters to "interesting" node types:
    │     function_definition, class_definition, method_definition,
    │     impl_item (Rust), function_item (Rust), …
    ▼
list[ASTNode]    ← our serialisable model
```

**`ASTNode` model**:
```python
class ASTNode(BaseModel):
    type: str         # e.g. "function_definition"
    name: str         # extracted identifier, e.g. "add"
    start_byte: int   # byte offset in source
    end_byte: int
    start_line: int   # 1-indexed
    end_line: int
    depth: int        # depth from AST root
```

#### Why byte offsets?

Tree-sitter reports positions in **bytes** (not characters).  This is
important for multibyte Unicode: `"café"` is 4 chars but 5 bytes in UTF-8.
We always use byte offsets for slicing, then decode to str.

#### Error tolerance

Tree-sitter **always returns a tree**, even for syntactically invalid code.
Parse errors are recorded as `ERROR` or `MISSING` nodes and collected in
`ParsedFile.parse_errors`.  The pipeline continues with whatever valid nodes
were found — partial indexing is better than no indexing.

---

### Stage 3 — Chunking

A **chunk** is the fundamental unit of indexing.  It is a piece of code
with a stable ID, language tag, and byte/line offsets — everything needed
to retrieve it and display it to users.

#### AST Chunker (default, recommended)

```
list[ASTNode]
    │
    ▼  For each AST node:
    │  1. Slice source_bytes[node.start_byte : node.end_byte]
    │  2. Map node.type → ChunkType (function/class/method/…)
    │  3. Extract identifier name from child nodes
    │  4. Apply min/max size guards
    │  5. Attach context_before / context_after lines
    ▼
list[Chunk]       ← one chunk per function / class / method
```

Example: given this Python file:

```python
def add(a, b):       # → Chunk(type=FUNCTION, name="add", lines=1-2)
    return a + b

class Calculator:    # → Chunk(type=CLASS, name="Calculator", lines=4-8)
    def mul(self, x, y):   # → Chunk(type=METHOD, name="mul", lines=5-7)
        return x * y
```

The AST chunker produces **3 chunks**: `add`, `Calculator`, and `mul`.
`Calculator` includes its body (including `mul`) so its content overlaps.

#### Token Chunker

```
source lines
    │
    ▼  accumulate lines until token budget exhausted
    │  → flush as chunk
    │  → carry over chunk_overlap tokens
    ▼
list[Chunk]      ← token-bounded, language-agnostic
```

Token counting uses `tiktoken` (`cl100k_base` encoding, matches GPT-4).
This chunker is best for **unknown or unsupported languages** and for
producing chunks of predictable size for fixed-context models.

#### Sliding Window Chunker

Character-based windows.  The simplest possible strategy.

```
content = "A" * 1000
chunk_size = 200, overlap = 20
→ windows at [0:200], [180:380], [360:560], …
```

Optionally snaps boundaries to the nearest newline to avoid mid-token splits.

---

### Stage 4 — Embedding

Each chunk is converted to a dense float vector.  The vector captures the
**semantic meaning** of the code so similar code has similar vectors.

```
list[Chunk]
    │
    ▼  chunk.to_embedding_text()
    │  Builds a contextual header:
    │  "[python] [function] src/auth.py :: verify_token"
    │  + raw chunk content
    │
    ▼  embedder.embed_texts(texts)   ← batched for efficiency
    │
    ▼
list[EmbeddedChunk]
    ├── chunk (original)
    ├── embedding: list[float]   ← e.g. 384 dims for MiniLM
    └── model: str
```

The contextual header is crucial: without it, the embedding model sees only
code tokens without knowing the language, path, or semantic type.  With it,
queries like "parse token in Go" will rank Go functions above Python ones.

---

### Stage 5 — Vector Store

```
list[EmbeddedChunk]
    │
    ▼  vector_store.upsert(embedded_chunks)
    │  • Each chunk becomes one "point" / "document"
    │  • Vector is indexed in an ANN structure (HNSW for Qdrant)
    │  • Metadata is stored alongside for filtering
    ▼
Persistent index (ChromaDB file / Qdrant server / RAM)
```

**Upsert semantics**: if a chunk with the same `id` already exists, it is
replaced.  Since chunk IDs are deterministic (SHA-256 of `file_id:start_byte:end_byte`),
re-indexing the same content is idempotent.

---

### Stage 6 — Retrieval

```
query: "function that authenticates a JWT token"
    │
    ▼  embedder.embed_texts([query])
    │  → query_vector: list[float]
    │
    ▼  vector_store.query(query_vector, top_k=10, filters={...})
    │  → cosine similarity ANN search
    │
    ▼  (optional) CrossEncoder.predict([(query, chunk.content), ...])
    │  → reranked by cross-encoder score
    │
    ▼
SearchResponse(results=[SearchResult(chunk, score, rank), ...])
```

**Two-stage retrieval** (bi-encoder + cross-encoder):

```
Stage 1: Bi-encoder (fast, recall-oriented)
  embed(query) → ANN search → top-20 candidates   ← O(log N)

Stage 2: Cross-encoder (accurate, precision-oriented)
  cross_encoder(query, doc₁), …, cross_encoder(query, doc₂₀)
  → rerank → top-5 final results                   ← O(k) forward passes
```

Cross-encoder reranking is optional (adds ~100-200ms on CPU) but significantly
improves precision for code-specific queries.

---

## Data Flow Diagram

```
                     ┌─────────────────────────────────┐
                     │         INDEXING PATH            │
                     └────────────────┬────────────────┘
                                      │
  ┌──────────────────────────────────────────────────────────────────┐
  │                                                                  │
  │  ./myrepo/                                                       │
  │     src/auth.py         CodebaseWalker                          │
  │     src/api.py      ──────────────────▶  SourceFile[]           │
  │     tests/test.py        (walks, filters,                       │
  │     ...                   detects lang)                          │
  │                                │                                 │
  │                                ▼                                 │
  │                       TreeSitterParser                           │
  │                   ─────────────────────▶  ParsedFile            │
  │                    (Tree-sitter AST,         (with ASTNode[])   │
  │                     error tolerant)                              │
  │                                │                                 │
  │                                ▼                                 │
  │                   ┌────────────────────┐                        │
  │                   │    Chunker         │                        │
  │                   │  ┌─────────────┐  │                        │
  │                   │  │ ASTChunker  │  │                        │
  │                   │  │ TokenChunker│  │──▶  Chunk[]            │
  │                   │  │ SlidingWin  │  │                        │
  │                   │  └─────────────┘  │                        │
  │                   └────────────────────┘                        │
  │                                │                                 │
  │                                ▼                                 │
  │                   ┌────────────────────┐                        │
  │                   │    Embedder        │                        │
  │                   │  ┌─────────────┐  │                        │
  │                   │  │ OpenAI      │  │                        │
  │                   │  │ SentenceTrf │  │──▶  EmbeddedChunk[]    │
  │                   │  │ Ollama      │  │      (vec: float[384]) │
  │                   │  └─────────────┘  │                        │
  │                   └────────────────────┘                        │
  │                                │                                 │
  │                                ▼                                 │
  │                   ┌────────────────────┐                        │
  │                   │   Vector Store     │                        │
  │                   │  ┌─────────────┐  │                        │
  │                   │  │ ChromaDB    │  │                        │
  │                   │  │ Qdrant      │  │◀──▶  Persistent Index  │
  │                   │  │ In-Memory   │  │                        │
  │                   │  └─────────────┘  │                        │
  │                   └────────────────────┘                        │
  │                                                                  │
  └──────────────────────────────────────────────────────────────────┘

                     ┌─────────────────────────────────┐
                     │         RETRIEVAL PATH           │
                     └────────────────┬────────────────┘
                                      │
  ┌──────────────────────────────────────────────────────────────────┐
  │                                                                  │
  │  "parse JWT token"                                               │
  │       │                                                          │
  │       ▼                                                          │
  │  Embedder.embed_texts(["parse JWT token"])                       │
  │       │  → [0.12, -0.34, 0.89, …]  (384 dims)                  │
  │       ▼                                                          │
  │  VectorStore.query(vec, top_k=20)                               │
  │       │  ANN search via HNSW / brute force                       │
  │       ▼                                                          │
  │  SearchResult[] ──▶ (optional) CrossEncoder rerank               │
  │       │                                                          │
  │       ▼                                                          │
  │  SearchResponse(results=[                                         │
  │    {chunk: "def verify_jwt(token):", score: 0.91, rank: 1},     │
  │    {chunk: "func ParseToken(t string)", score: 0.87, rank: 2},  │
  │    …                                                             │
  │  ])                                                              │
  └──────────────────────────────────────────────────────────────────┘
```

---

## Module Map

```
src/code_indexer/
│
├── core/
│   ├── models.py          ← All Pydantic data models
│   │   ├── SourceFile     – raw file with content + metadata
│   │   ├── ASTNode        – serialisable Tree-sitter node
│   │   ├── ParsedFile     – source + extracted AST nodes
│   │   ├── Chunk          – indexable code unit
│   │   ├── EmbeddedChunk  – chunk + float[] vector
│   │   ├── SearchQuery    – query parameters
│   │   ├── SearchResult   – chunk + score + rank
│   │   ├── SearchResponse – full query response
│   │   └── IndexStats     – collection statistics
│   │
│   ├── config.py          ← Pydantic-Settings configuration
│   │   ├── ParserSettings
│   │   ├── ChunkerSettings
│   │   ├── EmbeddingSettings
│   │   ├── VectorStoreSettings
│   │   ├── APISettings
│   │   └── Settings       – root config, get_settings() singleton
│   │
│   └── exceptions.py      ← Custom exception hierarchy
│       ├── CodeIndexerError   – base
│       ├── ParserError
│       ├── ChunkerError
│       ├── EmbeddingError
│       ├── VectorStoreError
│       └── IndexerError
│
├── parsers/
│   ├── base.py            ← BaseParser ABC
│   ├── tree_sitter_parser.py  ← TreeSitterParser implementation
│   └── language_registry.py  ← Extension→Language map + grammar loader
│
├── chunkers/
│   ├── base.py            ← BaseChunker ABC + shared helpers
│   ├── ast_chunker.py     ← ASTChunker (Tree-sitter node boundaries)
│   ├── token_chunker.py   ← TokenChunker (tiktoken)
│   └── sliding_window.py  ← SlidingWindowChunker (character-based)
│
├── embeddings/
│   ├── base.py            ← BaseEmbedder ABC
│   ├── openai_embedder.py      ← OpenAI Embeddings API
│   ├── sentence_transformer_embedder.py  ← Local HuggingFace model
│   └── ollama_embedder.py      ← Ollama HTTP API
│
├── vectorstore/
│   ├── base.py            ← BaseVectorStore ABC
│   ├── chroma_store.py    ← ChromaDB (local persistent)
│   ├── qdrant_store.py    ← Qdrant (production-grade)
│   └── in_memory_store.py ← NumPy brute-force (tests / small repos)
│
├── indexer/
│   ├── walker.py          ← CodebaseWalker (filesystem traversal)
│   └── pipeline.py        ← IndexingPipeline (orchestrator)
│                              build_chunker() / build_embedder() / build_vector_store()
│
├── retrieval/
│   ├── retriever.py       ← CodeRetriever (embed query → ANN search)
│   └── reranker.py        ← BaseReranker, ScoreReranker, CrossEncoderReranker
│
├── api/
│   ├── app.py             ← FastAPI factory, lifespan, exception handlers
│   ├── middleware.py      ← APIKeyMiddleware, RequestLoggingMiddleware
│   └── routes/
│       ├── health.py      ← GET /health, GET /ping
│       ├── index.py       ← POST /index/directory, POST /index/file, …
│       └── search.py      ← POST /search, POST /search/rerank
│
└── cli/
    └── main.py            ← click CLI: index, search, stats, serve
```

---

## Quick Start

### 1. Install

```bash
git clone <repo>
cd code-indexer

# Minimal install (local sentence-transformers + ChromaDB)
pip install -e .

# Development install
pip install -e ".[dev]"
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env — the defaults work out of the box for local usage.
```

### 3. Index a codebase

```bash
# Index the current directory
code-indexer index .

# Index a specific project with a clean slate
code-indexer index ~/projects/myapp --clear

# Index with extra metadata tags
code-indexer index . --metadata branch=main --metadata team=backend
```

**What happens during indexing:**

```
1. Walker counts 247 Python/JS/TS files (skipping node_modules, .git, …)
2. Each file is parsed with Tree-sitter → AST nodes extracted
3. ASTChunker produces 1,832 chunks (functions + classes)
4. Chunks are embedded in batches of 64 → 1,832 vectors (384 dims each)
5. Vectors are upserted into ChromaDB (.chroma/ directory)
6. Total: 47 seconds on a MacBook M2 (cold model load included)
```

### 4. Search

```bash
# Natural language search
code-indexer search "function that validates user tokens"

# Filter by language and chunk type
code-indexer search "database connection pool" --language python --type function

# Include surrounding context in output
code-indexer search "error handler" --context --top-k 3

# Output raw JSON (useful for scripting)
code-indexer search "parse config" --json-output | jq '.results[0].chunk.path'
```

### 5. Start the API server

```bash
code-indexer serve
# API available at http://localhost:8000
# Swagger UI at   http://localhost:8000/docs
```

### 6. Use the Python API directly

```python
from code_indexer.core.config import get_settings
from code_indexer.indexer.pipeline import IndexingPipeline
from code_indexer.core.models import SearchQuery

# Index
pipeline = IndexingPipeline()
result = pipeline.index_directory("./my-project")
print(f"Indexed {result.chunks_stored} chunks in {result.elapsed_seconds:.1f}s")

# Search
from code_indexer.indexer.pipeline import build_embedder, build_vector_store
from code_indexer.retrieval.retriever import CodeRetriever

settings = get_settings()
retriever = CodeRetriever(
    embedder=build_embedder(settings),
    vector_store=build_vector_store(settings),
)

response = retriever.search(SearchQuery(
    query="function that parses configuration files",
    top_k=5,
    language="python",
))

for r in response.results:
    print(f"  [{r.score:.2f}] {r.chunk.path}:{r.chunk.start_line} — {r.chunk.name}")
```

---

## Configuration Reference

All settings are read from environment variables or a `.env` file.
See `.env.example` for the complete list with comments.

### Parser Settings (`PARSER_` prefix)

| Variable | Default | Description |
|---|---|---|
| `PARSER_MAX_FILE_SIZE_BYTES` | `1048576` | Skip files larger than this (1 MB) |
| `PARSER_CONTEXT_LINES` | `3` | Lines of surrounding code per chunk |
| `PARSER_IGNORE_PATTERNS` | `[node_modules/**, .git/**, ...]` | Glob patterns to skip |

### Chunker Settings (`CHUNKER_` prefix)

| Variable | Default | Options |
|---|---|---|
| `CHUNKER_STRATEGY` | `ast` | `ast`, `token`, `sliding_window` |
| `CHUNKER_CHUNK_SIZE` | `512` | Tokens (token/ast) or chars (sliding_window) |
| `CHUNKER_CHUNK_OVERLAP` | `64` | Units of overlap between consecutive chunks |
| `CHUNKER_MIN_CHUNK_SIZE` | `20` | Minimum word count to keep a chunk |
| `CHUNKER_MAX_CHUNK_SIZE` | `2048` | Hard upper bound; triggers fallback splitting |

### Embedding Settings (`EMBEDDING_` prefix)

| Variable | Default | Options |
|---|---|---|
| `EMBEDDING_PROVIDER` | `sentence_transformer` | `sentence_transformer`, `openai`, `ollama` |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Any valid model for the provider |
| `EMBEDDING_DIMENSIONS` | `384` | Output vector size (must match model) |
| `EMBEDDING_BATCH_SIZE` | `64` | Texts per embedding call |
| `EMBEDDING_OPENAI_API_KEY` | `` | Required for `openai` provider |
| `EMBEDDING_OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |

### Vector Store Settings (`VECTORSTORE_` prefix)

| Variable | Default | Options |
|---|---|---|
| `VECTORSTORE_PROVIDER` | `chroma` | `chroma`, `qdrant`, `in_memory` |
| `VECTORSTORE_COLLECTION_NAME` | `code_index` | Logical namespace for this index |
| `VECTORSTORE_CHROMA_PERSIST_DIR` | `.chroma` | Directory for ChromaDB data |
| `VECTORSTORE_QDRANT_URL` | `http://localhost:6333` | Qdrant server URL |
| `VECTORSTORE_QDRANT_API_KEY` | `` | Qdrant Cloud API key |
| `VECTORSTORE_DISTANCE_METRIC` | `cosine` | `cosine`, `dot`, `euclidean` |

### API Settings (`API_` prefix)

| Variable | Default | Description |
|---|---|---|
| `API_HOST` | `0.0.0.0` | Bind address |
| `API_PORT` | `8000` | TCP port |
| `API_WORKERS` | `1` | uvicorn worker count |
| `API_RELOAD` | `false` | Hot reload (dev only) |
| `API_KEY` | `` | Bearer token auth (empty = disabled) |
| `API_LOG_LEVEL` | `info` | `debug`, `info`, `warning`, `error` |

---

## API Reference

### Health

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness probe — always 200 while process is running |
| `GET` | `/ping` | Minimal ping for load balancers |

### Indexing

| Method | Path | Description |
|---|---|---|
| `POST` | `/index/directory` | Index an entire directory tree |
| `POST` | `/index/file` | Index a single file |
| `GET` | `/index/stats` | Return index statistics |
| `DELETE` | `/index/file` | Remove a file's chunks |
| `DELETE` | `/index/clear` | Wipe the entire index |

**POST /index/directory**

```json
{
  "path": "/absolute/path/to/codebase",
  "clear_existing": false,
  "extra_metadata": {
    "branch": "main",
    "repo": "myapp"
  }
}
```

Response:
```json
{
  "files_processed": 247,
  "files_skipped": 12,
  "chunks_produced": 1832,
  "chunks_embedded": 1832,
  "chunks_stored": 1832,
  "elapsed_seconds": 47.3,
  "errors": []
}
```

### Search

| Method | Path | Description |
|---|---|---|
| `POST` | `/search` | Bi-encoder vector similarity search |
| `POST` | `/search/rerank` | Bi-encoder + cross-encoder reranking |

**POST /search**

```json
{
  "query": "function that validates JWT tokens",
  "top_k": 5,
  "language": "python",
  "chunk_types": ["function", "method"],
  "path_prefix": "src/auth/",
  "min_score": 0.3,
  "include_context": true,
  "metadata_filters": {
    "branch": "main"
  }
}
```

Response:
```json
{
  "query": "function that validates JWT tokens",
  "results": [
    {
      "chunk": {
        "id": "a1b2c3d4...",
        "path": "src/auth/jwt.py",
        "language": "python",
        "chunk_type": "function",
        "name": "verify_token",
        "content": "def verify_token(token: str) -> bool:\n    ...",
        "start_line": 42,
        "end_line": 58,
        "context_before": "# JWT utilities",
        "context_after": ""
      },
      "score": 0.912,
      "rank": 1
    }
  ],
  "total": 5,
  "latency_ms": 23.4
}
```

---

## CLI Reference

```
code-indexer [OPTIONS] COMMAND [ARGS]...

Commands:
  index    Index a codebase directory
  search   Search the indexed codebase
  stats    Show index statistics
  serve    Start the API server
```

### `code-indexer index`

```
Usage: code-indexer index [OPTIONS] PATH

  Index a codebase directory.

Arguments:
  PATH  Root directory to index.

Options:
  --strategy TEXT        Chunking strategy (ast/token/sliding_window)
  --clear                Wipe existing index before indexing
  --provider TEXT        Embedding provider override
  --store TEXT           Vector store provider override
  --metadata KEY=VALUE   Extra metadata (repeatable)
```

### `code-indexer search`

```
Usage: code-indexer search [OPTIONS] QUERY

  Search the indexed codebase.

Arguments:
  QUERY  Natural language or code query.

Options:
  --top-k INTEGER       Number of results [default: 5]
  --language TEXT       Filter by language
  --type TEXT           Filter by chunk type (function/class/method/…)
  --min-score FLOAT     Minimum similarity score [default: 0.0]
  --context             Include surrounding context lines
  --json-output         Output raw JSON
```

---

## Chunking Strategies Explained

### Decision Flowchart

```
Do you have Tree-sitter support for the language?
    │
    ├─ YES → Is the codebase < 100k chunks?
    │            │
    │            ├─ YES → Use ASTChunker (best precision)
    │            └─ NO  → Use ASTChunker + TokenChunker fallback
    │
    └─ NO  → Do you need predictable token counts?
                 │
                 ├─ YES → Use TokenChunker
                 └─ NO  → Use SlidingWindowChunker
```

### Comparison

| Strategy | Boundary | Pros | Cons |
|---|---|---|---|
| `ast` | Function/class | Semantically meaningful, name-tagged | Requires grammar, variable size |
| `token` | Token count | Predictable size, works everywhere | May split mid-function |
| `sliding_window` | Character count | Zero dependencies, fastest | Least semantic, no type info |

---

## Embedding Backend Comparison

| Provider | Model | Dims | Speed | Quality | Cost | Privacy |
|---|---|---|---|---|---|---|
| `sentence_transformer` | `all-MiniLM-L6-v2` | 384 | Fast | Good | Free | Local |
| `sentence_transformer` | `microsoft/codebert-base` | 768 | Moderate | Excellent for code | Free | Local |
| `openai` | `text-embedding-3-small` | 1536 | Network-bound | Excellent | ~$0.02/1M tokens | API |
| `openai` | `text-embedding-3-large` | 3072 | Network-bound | Best | ~$0.13/1M tokens | API |
| `ollama` | `nomic-embed-text` | 768 | Moderate | Good | Free | Local |

---

## Vector Store Backend Comparison

| Provider | Setup | Scale | Persistence | Filtering | Recommended for |
|---|---|---|---|---|---|
| `in_memory` | None | < 50k chunks | None (RAM only) | Basic | Tests, quick demos |
| `chroma` | None (file) | < 1M chunks | File-based | Metadata | Development, small teams |
| `qdrant` | Docker/Cloud | Billions | File/Distributed | Full | Production |

---

## RAG Integration Guide

### With LangChain

```python
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

from code_indexer.core.config import get_settings
from code_indexer.core.models import SearchQuery
from code_indexer.indexer.pipeline import build_embedder, build_vector_store
from code_indexer.retrieval.retriever import CodeRetriever


class CodeIndexerRetriever:
    """Wrap CodeRetriever as a LangChain-compatible retriever."""

    def __init__(self, top_k: int = 5):
        settings = get_settings()
        self._retriever = CodeRetriever(
            embedder=build_embedder(settings),
            vector_store=build_vector_store(settings),
        )
        self._top_k = top_k

    def get_relevant_documents(self, query: str) -> list[Document]:
        response = self._retriever.search(SearchQuery(query=query, top_k=self._top_k))
        return [
            Document(
                page_content=r.chunk.content,
                metadata={
                    "path": r.chunk.path,
                    "language": r.chunk.language.value,
                    "name": r.chunk.name,
                    "score": r.score,
                },
            )
            for r in response.results
        ]


# Build a code Q&A chain
retriever = CodeIndexerRetriever(top_k=5)
llm = ChatOpenAI(model="gpt-4o")

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
)

result = qa_chain.invoke("How does the authentication system work?")
print(result["result"])
```

### Direct OpenAI API Integration

```python
import openai
from code_indexer.core.config import get_settings
from code_indexer.core.models import SearchQuery
from code_indexer.indexer.pipeline import build_embedder, build_vector_store
from code_indexer.retrieval.retriever import CodeRetriever

settings = get_settings()
retriever = CodeRetriever(
    embedder=build_embedder(settings),
    vector_store=build_vector_store(settings),
)

def code_qa(question: str) -> str:
    # 1. Retrieve relevant code chunks
    response = retriever.search(SearchQuery(query=question, top_k=5, include_context=True))

    # 2. Build context string
    context_parts = []
    for r in response.results:
        chunk = r.chunk
        context_parts.append(
            f"File: {chunk.path} (lines {chunk.start_line}-{chunk.end_line})\n"
            f"```{chunk.language.value}\n{chunk.content}\n```"
        )
    context = "\n\n---\n\n".join(context_parts)

    # 3. Call LLM with retrieved context
    client = openai.OpenAI()
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You are a code assistant. Answer questions about the codebase "
                           "using ONLY the provided code snippets.",
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            },
        ],
    )
    return completion.choices[0].message.content
```

---

## Deployment Guide

### Local Development

```bash
# Install
pip install -e ".[dev]"

# Start with hot reload
code-indexer serve --reload

# Or with make
make serve
```

### Docker (Single Container)

```bash
# Build the image
docker build -f docker/Dockerfile -t code-indexer:latest .

# Run with ChromaDB (default, file-based persistence)
docker run -p 8000:8000 \
  -v $(pwd)/.chroma:/app/.chroma \
  -v /path/to/myrepo:/code:ro \
  --env-file .env \
  code-indexer:latest
```

### Docker Compose (with Qdrant)

```bash
# Start API + Qdrant
docker compose -f docker/docker-compose.yml --profile qdrant up --build

# Index your codebase via the API
curl -X POST http://localhost:8000/index/directory \
  -H "Content-Type: application/json" \
  -d '{"path": "/code", "clear_existing": true}'
```

### Kubernetes (Production)

Key considerations for Kubernetes deployment:

1. **Vector store**: Use Qdrant Cloud or a Qdrant StatefulSet.  Do NOT use
   ChromaDB in a multi-replica setup (it's file-based, not distributed).

2. **Embedding model**: Use OpenAI or a dedicated embedding service.  Loading
   sentence-transformers inside each pod wastes memory.

3. **Horizontal scaling**: The API is stateless (state lives in the vector store
   and embedding service).  You can run multiple replicas.

4. **Indexing**: Run indexing as a Kubernetes Job (one-off) or CronJob
   (scheduled re-indexing), not as part of the API deployment.

```yaml
# Example: Kubernetes Deployment for the API
apiVersion: apps/v1
kind: Deployment
metadata:
  name: code-indexer-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: code-indexer-api
  template:
    spec:
      containers:
        - name: api
          image: code-indexer:latest
          ports:
            - containerPort: 8000
          env:
            - name: EMBEDDING_PROVIDER
              value: openai
            - name: EMBEDDING_OPENAI_API_KEY
              valueFrom:
                secretKeyRef:
                  name: openai-secret
                  key: api-key
            - name: VECTORSTORE_PROVIDER
              value: qdrant
            - name: VECTORSTORE_QDRANT_URL
              value: http://qdrant.qdrant.svc.cluster.local:6333
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
```

---

## Development Guide

### Project Setup

```bash
# Clone and install
git clone <repo>
cd code-indexer
pip install -e ".[dev]"
pre-commit install
```

### Running Tests

```bash
# All tests
make test

# Fast (no API tests)
make test-fast

# Single test file
pytest tests/test_chunkers/test_token_chunker.py -v
```

### Code Quality

```bash
make lint        # ruff check
make fmt         # ruff format
make type-check  # mypy
```

### Adding a New Language

1. Install the Tree-sitter grammar package:
   ```bash
   pip install tree-sitter-kotlin
   ```

2. Register it in `parsers/language_registry.py`:
   ```python
   # In _EXT_MAP
   ".kt": Language.KOTLIN,

   # In _GRAMMAR_MODULES
   Language.KOTLIN: "tree_sitter_kotlin",
   ```

3. Add the enum value to `core/models.py`:
   ```python
   class Language(str, Enum):
       KOTLIN = "kotlin"
   ```

4. Add relevant AST node types to `parsers/tree_sitter_parser.py`:
   ```python
   _INTERESTING_NODE_TYPES = frozenset({
       ...,
       "function_declaration",   # Kotlin
       "class_declaration",      # Kotlin
   })
   ```

### Adding a New Embedding Backend

1. Create `embeddings/my_embedder.py`:
   ```python
   from code_indexer.embeddings.base import BaseEmbedder

   class MyEmbedder(BaseEmbedder):
       def embed_texts(self, texts: list[str]) -> list[list[float]]:
           # your implementation
           ...
   ```

2. Wire it up in `indexer/pipeline.py` `build_embedder()`.

3. Add it to `EmbeddingSettings.provider` in `core/config.py`.

### Adding a New Vector Store

1. Create `vectorstore/my_store.py` implementing `BaseVectorStore`.
2. Wire it up in `build_vector_store()`.
3. Add settings to `VectorStoreSettings`.

---

## Glossary

| Term | Definition |
|---|---|
| **AST** | Abstract Syntax Tree — a tree representation of source code structure produced by a parser. |
| **Chunk** | A single indexable unit of code: one function, class, or text window. |
| **Embedding** | A dense float vector representing the semantic meaning of a piece of text. |
| **HNSW** | Hierarchical Navigable Small World — graph-based ANN index algorithm used by Qdrant. |
| **ANN** | Approximate Nearest Neighbour — finding similar vectors efficiently without exact search. |
| **RAG** | Retrieval-Augmented Generation — LLM pattern that retrieves relevant context before generating. |
| **Bi-encoder** | Model that embeds query and document independently; fast but less accurate than cross-encoder. |
| **Cross-encoder** | Model that jointly processes query + document; more accurate but O(k) inference cost. |
| **Upsert** | Insert-or-update: if the record exists (by ID), replace it; otherwise insert it. |
| **Tree-sitter** | Incremental parser generator that produces CSTs for 100+ languages. |

---

## License

MIT — see [LICENSE](LICENSE) for details.
