# Code Indexer

> **Production-ready codebase indexing system for RAG, impact analysis, and code generation.**
>
> Parse your entire codebase with Tree-sitter, extract a structural knowledge
> graph (call graph, inheritance, dependency injection), enrich identifiers with
> LLM-generated definitions, and query everything in natural language — all from
> a clean Python library, REST API, or CLI.

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
   - [Stage 7 — Graph Extraction](#stage-7--graph-extraction)
   - [Stage 8 — Semantic Enrichment](#stage-8--semantic-enrichment)
4. [Data Flow Diagram](#data-flow-diagram)
5. [Module Map](#module-map)
6. [Quick Start](#quick-start)
7. [Configuration Reference](#configuration-reference)
8. [API Reference](#api-reference)
9. [CLI Reference](#cli-reference)
10. [Chunking Strategies Explained](#chunking-strategies-explained)
11. [Embedding Backend Comparison](#embedding-backend-comparison)
12. [Vector Store Backend Comparison](#vector-store-backend-comparison)
13. [Graph Store Backend Comparison](#graph-store-backend-comparison)
14. [RAG Integration Guide](#rag-integration-guide)
15. [Deployment Guide](#deployment-guide)
16. [Development Guide](#development-guide)

---

## Why Code Indexer?

| Problem | Code Indexer Solution |
|---|---|
| LLMs forget code context > 200k tokens | Semantic search retrieves only the relevant chunks |
| Naive line-splitting breaks functions in half | AST-aware chunking respects function/class boundaries |
| Different codebases use different languages | Tree-sitter grammars for 8+ languages out of the box |
| Expensive GPU embedding runs | Local `sentence-transformers` or remote OpenAI/Ollama |
| Locked into one vector DB | Pluggable: ChromaDB, Qdrant, or in-memory |
| "What breaks if I change X?" is unanswerable | Call-graph + inheritance + INJECTS edges → impact analysis |
| DI dependencies invisible to the call graph | `INJECTS` edges capture constructor-injected types |
| Abbreviations like `req`/`tx` miss semantic search | `TermEnricher` LLM-expands identifiers before indexing |
| Flat ranked lists hide cross-file structure | `ConcernClusterer` groups results into named themes |
| Complex setup required | One command: `code-indexer index ./myrepo` |

---

## Architecture Overview

The system is built as two complementary pipelines sharing the same
Tree-sitter parsing layer, each with a clean abstract interface and
multiple interchangeable backends.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          CODE INDEXER SYSTEM                            │
│                                                                         │
│   ┌────────────┐    ┌──────────────────────────────────────────────┐   │
│   │  CLI / API │    │              Indexing Pipelines               │   │
│   │  Layer     │───▶│                                              │   │
│   └────────────┘    │  ┌─────────────────┐  ┌────────────────────┐│   │
│                     │  │  Vector Pipeline │  │   Graph Pipeline   ││   │
│                     │  │                 │  │                    ││   │
│                     │  │ Parser→Chunker  │  │ Parser→GraphExtract││   │
│                     │  │ →Embedder       │  │ →TermEnricher      ││   │
│                     │  │ →VectorStore    │  │ →GraphStore        ││   │
│                     │  └────────┬────────┘  └─────────┬──────────┘│   │
│                     └──────────────────────────────────────────────┘   │
│                               │                        │               │
│                               ▼                        ▼               │
│                     ┌──────────────────┐    ┌─────────────────────┐   │
│                     │   Vector Store   │    │    Graph Store      │   │
│                     │ ChromaDB/Qdrant/ │    │  in-memory / Neo4j  │   │
│                     │ in-memory        │    │  5 edge types       │   │
│                     └──────────────────┘    └─────────────────────┘   │
│                               │                        │               │
│                               └───────────┬────────────┘               │
│                                           ▼                            │
│                                ┌─────────────────────┐                 │
│                                │  HybridRetriever    │                 │
│                                │  vector + graph     │                 │
│                                │  + Katz centrality  │                 │
│                                │  + ContextPacker    │                 │
│                                │  + ConcernClusterer │                 │
│                                └─────────────────────┘                 │
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
| **Graph Extractor** | Extract CALLS / IMPORTS / INHERITS_FROM / INJECTS edges | `graph/pipeline.py` |
| **Graph Store** | Store nodes + edges, run traversal queries | `graph/in_memory_graph.py`, `graph/neo4j_store.py` |
| **TermEnricher** | LLM-expand identifiers, generate definitions | `graph/term_enricher.py` |
| **ConcernClusterer** | Group retrieved results into named themes | `graph/concern_clusterer.py` |
| **ContextPacker** | Knapsack-optimal symbol selection for token budget | `graph/context_packer.py` |
| **Retriever** | Vector + graph hybrid query with Katz scoring | `retrieval/retriever.py` |
| **API** | FastAPI REST endpoints (vector + graph) | `api/app.py`, `api/routes/` |
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

### Stage 7 — Graph Extraction

Runs in parallel with the vector pipeline (or on its own via `code-indexer graph index`).
The `GraphExtractor` walks the same Tree-sitter AST and builds a typed property graph.

```
ParsedFile (AST)
    │
    ▼  GraphExtractor.extract(parsed_file)
    │
    ├── FileNode          {id, path, language, sha256}
    │
    ├── SymbolNode[]      {id, name, qualified_name, kind,
    │                      file_path, start_line, end_line, decorators}
    │
    └── GraphEdge[]
            DEFINES        File → Symbol
            CONTAINS       Class → Method
            IMPORTS        File → File  {module_string, is_relative}
            CALLS          Symbol → Symbol  {line, count}
            INHERITS_FROM  Symbol → Symbol
            INJECTS        Symbol → Symbol  {field_name, field_type}
```

**Call resolution** uses a three-tier name matcher:

```
1. Exact qualified name match   → confidence 1.0
2. Unqualified name match       → confidence 0.8
3. Suffix / fuzzy match         → confidence 0.6
```

Edges below the `min_confidence` threshold are dropped before storage.

**INJECTS edges** are extracted from constructor parameters and typed class
fields (Python `__init__`, TypeScript constructor, Java `@Autowired`) — capturing
dependency-injection relationships that pure call graphs miss.

---

### Stage 8 — Semantic Enrichment

Runs offline once per codebase.  All results are persisted to `.term_kb.json`
so enrichment cost is paid once and amortised across all future queries.

```
SymbolNode[]
    │
    ▼  split_identifier(sym.name)
    │  getUserByEmail → ["get", "user", "by", "email"]
    │
    ▼  TermEnricher (LLM: cheap model, e.g. gpt-4o-mini)
    │  For each short/abbreviated token:
    │    1. Expand:  "req" → "HTTP request"   (context-aware)
    │    2. Define:  "HTTP request" → "Structured message sent by a client to a server"
    │
    ▼  EnrichedTerm[]  stored in TermKnowledgeBase
    │
    ▼  TermChunkExtractor (LLM: cheap model)
    │  For each (symbol, term) pair:
    │    "Describe only what handle_payment does with rate limiting"
    │    → focused one-sentence summary
    │
    ▼  TermChunk[]  embedded and added to vector store
       (alongside whole-function chunks)
```

At query time, `ConcernClusterer` groups retrieved `TermChunk` results into
2–5 named concerns using a stronger model (e.g. `gpt-4o`), and
`render_concern_block()` prepends a soft-guidance map to the LLM prompt.

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
├── graph/
│   ├── models.py          ← Graph Pydantic models
│   │   ├── FileNode       – source file node
│   │   ├── SymbolNode     – function / class / method / … node
│   │   ├── DirectoryNode  – directory node
│   │   ├── GraphEdge      – typed directed edge (CALLS, IMPORTS, …)
│   │   ├── GraphSnapshot  – full in-memory graph + Katz centrality
│   │   ├── SymbolContext  – neighbourhood for one symbol + to_mermaid()
│   │   └── ImpactResult   – blast-radius result + to_mermaid()
│   │
│   ├── extractor.py       ← GraphExtractor: AST → nodes + edges
│   ├── pipeline.py        ← GraphIndexingPipeline, build_graph_store()
│   ├── in_memory_graph.py ← InMemoryGraphStore (BFS, Katz, impact)
│   ├── neo4j_store.py     ← Neo4jGraphStore (Cypher queries)
│   │
│   ├── term_utils.py      ← split_identifier()
│   ├── term_enricher.py   ← TermEnricher, EnrichedTerm, TermKnowledgeBase
│   ├── term_chunker.py    ← TermChunkExtractor, TermChunk
│   ├── concern_clusterer.py ← ConcernClusterer, Concern
│   └── context_packer.py  ← ContextPacker (knapsack), render_concern_block()
│
├── llm/
│   ├── base.py            ← BaseLLMClient ABC: complete(prompt) → str
│   ├── openai_client.py   ← OpenAILLMClient (gpt-4o-mini, gpt-4o, …)
│   ├── ollama_client.py   ← OllamaLLMClient (local, free)
│   └── __init__.py        ← make_llm_client(), make_cluster_llm_client()
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
│       ├── search.py      ← POST /search, POST /search/rerank
│       └── graph.py       ← POST /graph/index, GET /graph/stats,
│                              POST /graph/callers, POST /graph/callees,
│                              POST /graph/impact, POST /graph/context, …
│
└── cli/
    └── main.py            ← click CLI: index, search, stats, serve,
                               graph index, graph callers, graph callees,
                               graph context, graph stats
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

### 5. Build the knowledge graph

The graph pipeline runs independently of the vector pipeline and produces
structural edges (call graph, inheritance, dependency injection).

```bash
# Build the structural graph for the same codebase
code-indexer graph index .

# Summary output:
#   Files processed: 247
#   Symbols created: 1,543
#   Edges created:   4,891  (CALLS + IMPORTS + INHERITS_FROM + INJECTS)
#   Elapsed: 12.3s

# Ask who calls a function
code-indexer graph callers verify_token
#   src/api/routes/auth.py:88  auth.routes.auth.login_handler  [function]
#   src/middleware/auth.py:34  middleware.auth.require_auth     [function]

# Get the full structural context for RAG (also outputs Mermaid diagram)
code-indexer graph context verify_token

# Impact analysis — what breaks if I change this?
code-indexer graph callers verify_token --depth 3
```

### 6. Start the API server

```bash
code-indexer serve
# API available at http://localhost:8000
# Swagger UI at   http://localhost:8000/docs
```

### 7. Use the Python API directly

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

**Graph API:**

```python
from code_indexer.graph.pipeline import GraphIndexingPipeline, build_graph_store

settings = get_settings()
graph_store = build_graph_store(settings)

# Build the graph
pipeline = GraphIndexingPipeline(settings=settings, graph_store=graph_store)
result = pipeline.index_directory("./my-project")
print(f"Symbols: {result.symbols_created}  Edges: {result.edges_created}")

# Impact analysis — what breaks if verify_token changes?
syms = graph_store.find_symbols_by_name("verify_token")
impact = graph_store.get_impact_set(syms[0].id, min_confidence=0.6)
print(impact.to_mermaid())   # renders in GitHub / Notion / VS Code

# Structural context for an LLM prompt
ctx = graph_store.get_symbol_context(syms[0].id)
print(ctx.to_context_text())   # callers, callees, parent, imports, decorators
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

### Graph Settings (`GRAPH_` prefix)

| Variable | Default | Options |
|---|---|---|
| `GRAPH_PROVIDER` | `in_memory` | `in_memory`, `neo4j` |
| `GRAPH_NEO4J_URL` | `bolt://localhost:7687` | Neo4j Bolt URL |
| `GRAPH_NEO4J_USER` | `neo4j` | Neo4j username |
| `GRAPH_NEO4J_PASSWORD` | `` | Neo4j password |
| `GRAPH_MIN_CONFIDENCE` | `0.6` | Drop call edges below this score |

### LLM Settings (`LLM_` prefix)

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `openai` | `openai`, `ollama` |
| `LLM_MODEL` | `gpt-4o-mini` | Cheap model for bulk enrichment (expand, define, rank) |
| `LLM_CLUSTER_MODEL` | `gpt-4o` | Strong model for concern clustering (one call per query) |
| `LLM_API_KEY` | `` | OpenAI API key (required for `openai` provider) |
| `LLM_BASE_URL` | `http://localhost:11434` | Ollama server URL |

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

### Graph

| Method | Path | Description |
|---|---|---|
| `POST` | `/graph/index` | Build structural graph for a directory |
| `GET` | `/graph/stats` | Symbol / edge counts by type |
| `DELETE` | `/graph/clear` | Wipe the graph index |
| `GET` | `/graph/symbol/{id}` | Fetch one symbol by ID |
| `POST` | `/graph/symbol/search` | Find symbols by name |
| `GET` | `/graph/file/{id}/symbols` | All symbols in a file |
| `POST` | `/graph/callers` | BFS callers of a symbol |
| `POST` | `/graph/callees` | BFS callees of a symbol |
| `POST` | `/graph/call-path` | Shortest call path between two symbols |
| `POST` | `/graph/import-graph` | Transitive import closure for a file |
| `POST` | `/graph/subclasses` | Direct subclasses of a symbol |
| `POST` | `/graph/superclasses` | Inheritance chain above a symbol |
| `POST` | `/graph/context` | Full structural neighbourhood (+ Mermaid) |
| `POST` | `/graph/impact` | Blast-radius analysis (+ Mermaid) |

**POST /graph/impact** request:

```json
{
  "symbol_id": "a1b2c3...",
  "min_confidence": 0.6,
  "depth": 5
}
```

Response:

```json
{
  "symbol": { "name": "verify_token", "kind": "method", "file_path": "src/auth/jwt.py" },
  "direct_callers": [...],
  "transitive_callers": [...],
  "subclasses": [],
  "importing_files": ["src/api/routes/auth.py"],
  "affected_files": ["src/api/routes/auth.py", "src/middleware/auth.py"],
  "mermaid": "flowchart TD\n    target[\"◆ verify_token [method]  ← CHANGED\"]..."
}
```

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
  index    Index a codebase directory (vector pipeline)
  search   Search the indexed codebase
  stats    Show vector index statistics
  serve    Start the API server
  graph    Code knowledge graph commands (structural relationships)
    index    Build structural graph for a directory
    callers  Show what calls SYMBOL_NAME
    callees  Show what SYMBOL_NAME calls
    context  Full structural context for SYMBOL_NAME (for RAG)
    stats    Show graph statistics
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

### `code-indexer graph`

```
Usage: code-indexer graph COMMAND [ARGS]...

  Code knowledge graph commands (structural relationships).

Commands:
  index    Build the structural graph for a directory.
  callers  Show what calls SYMBOL_NAME.
  callees  Show what SYMBOL_NAME calls.
  context  Full structural context for SYMBOL_NAME (callers, callees, imports,
           decorators) — formatted for direct use in LLM prompts.
  stats    Show graph statistics (node / edge counts by type).
```

```bash
# Build graph (default: in-memory; for persistence use neo4j)
code-indexer graph index ./myrepo
code-indexer graph index ./myrepo --provider neo4j --clear

# Call graph traversal
code-indexer graph callers verify_token --depth 2
code-indexer graph callees handle_payment --depth 1 --json-output

# Structural context (includes Mermaid flowchart)
code-indexer graph context JWTHandler.verify_token
```

---

## Graph Store Backend Comparison

| Provider | Setup | Persistence | Scale | Recommended for |
|---|---|---|---|---|
| `in_memory` | None | RAM only | < 500k symbols | Development, CI, tests |
| `neo4j` | Docker / Neo4j AuraDB | File / Cloud | Millions of nodes | Production, large monorepos |

```bash
# Run Neo4j locally with Docker
docker run -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=none \
  neo4j:5

# Configure
GRAPH_PROVIDER=neo4j
GRAPH_NEO4J_URL=bolt://localhost:7687
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
| **CALLS edge** | Graph edge: function A calls function B (extracted from AST call expressions). |
| **IMPORTS edge** | Graph edge: file A imports file B (with module string, alias, is_relative metadata). |
| **INHERITS_FROM edge** | Graph edge: class A extends class B. |
| **INJECTS edge** | Graph edge: class A holds a typed reference to class B via constructor/field injection. |
| **Katz centrality** | Node score based on how many things transitively depend on it — finds architectural hotspots. |
| **Impact analysis** | "If I change X, what else might break?" — BFS over reversed CALLS/INHERITS_FROM/IMPORTS edges. |
| **TermEnricher** | LLM component that expands abbreviated identifiers (`req` → `HTTP request`) and generates definitions. |
| **TermChunk** | A focused LLM summary of one (function, term) pair — more precise than a whole-function chunk. |
| **ConcernClusterer** | Groups retrieved TermChunks into 2–5 named semantic concerns using a capable LLM. |
| **ContextPacker** | Knapsack-optimal selector that fits the best symbols within a token budget. |
| **Mermaid** | Text format rendered as flowchart diagrams by GitHub, Notion, VS Code, etc. |
| **Neo4j** | Graph database — stores nodes and edges, queried with Cypher (like SQL for graphs). |

---

## License

MIT — see [LICENSE](LICENSE) for details.
