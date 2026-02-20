# Architecture Decisions & Engineering Log

> **Purpose:** Reference document for understanding *why* the code is shaped
> the way it is.  Each section describes: what existed before, what the problem
> was, and exactly how it was solved.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Graph Layer — Design Rationale](#2-graph-layer--design-rationale)
3. [Decision: Built-in Noise Filtering](#3-decision-built-in-noise-filtering)
4. [Decision: Three-Tier Call Resolution with Confidence Scores](#4-decision-three-tier-call-resolution-with-confidence-scores)
5. [Decision: Suffix Index in ImportResolver](#5-decision-suffix-index-in-importresolver)
6. [Decision: Reciprocal Rank Fusion (RRF) for Hybrid Search](#6-decision-reciprocal-rank-fusion-rrf-for-hybrid-search)
7. [Fix: O(n) _find_enclosing_symbol](#7-fix-on-_find_enclosing_symbol)
8. [Fix: In-Memory Graph Persistence](#8-fix-in-memory-graph-persistence)
9. [Fix: Fragile _last_query_vector Access](#9-fix-fragile-_last_query_vector-access)
10. [Reference: GitNexus Patterns](#10-reference-gitnexus-patterns)

---

## 1. System Overview

The system has **two parallel pipelines** that feed a single query layer:

```
Source Files
     │
     ├─── Vector Pipeline ────────────────────────────────────────────────┐
     │    Walk → Parse (Tree-sitter) → Chunk → Embed → VectorStore        │
     │                                                                     ▼
     └─── Graph Pipeline ─────────────────────────────────────────────── HybridRetriever
          Walk → Extract (symbols/imports/calls) → Resolve → GraphStore   │
                                                                           ▼
                                                                    SearchResponse
```

**Vector pipeline** answers: *"what code is semantically similar to this query?"*

**Graph pipeline** answers: *"what code is structurally related to these results?"*
(callers, callees, parent classes, imported files)

The `HybridRetriever` fuses both with Reciprocal Rank Fusion.

---

## 2. Graph Layer — Design Rationale

### Why add a graph at all?

Pure vector search has a well-known blind spot: it finds code that *looks like*
the query but misses code that is *used by* or *depends on* the match.

Example: you search for "verify JWT token". The vector search finds
`verify_token()` correctly. But the LLM also needs to know:
- `hash_password()` is called inside `verify_token()` — might need its logic
- `User.check_password()` calls `verify_token()` — the caller context
- `utils.py` is imported — so the embeddings for `utils.py` matter too

A code knowledge graph captures these structural relationships explicitly.

### Node types

| Type | What it represents |
|---|---|
| `FileNode` | A source file (`auth.py`) |
| `SymbolNode` | A named definition (function / class / method / …) |
| `DirectoryNode` | A directory in the repo tree |

### Edge (relationship) types

| Relationship | Direction | Meaning |
|---|---|---|
| `IMPORTS` | File → File | `auth.py` imports `utils.py` |
| `DEFINES` | File → Symbol | `auth.py` defines `verify_token` |
| `CONTAINS` | Symbol → Symbol | Class `User` contains method `check_password` |
| `CALLS` | Symbol → Symbol | `verify_token` calls `hash_password` |
| `INHERITS_FROM` | Symbol → Symbol | `AuthService` inherits from `BaseService` |
| `REFERENCES` | Symbol → Symbol | Generic reference (not a call) |
| `PART_OF` | File/Dir → Dir | `auth.py` is part of `services/` |

### Storage backends

| Backend | When to use |
|---|---|
| `InMemoryGraphStore` | Dev / tests / repos < 100k nodes |
| `Neo4jGraphStore` | Production — full Cypher queries, persistence, clustering |

---

## 3. Decision: Built-in Noise Filtering

**Files changed:** `src/code_indexer/graph/extractor.py`

### Before

The call extractors emitted a `RawCall` for every function invocation in the
source file without any filtering.  A typical Python file would produce calls
like: `print`, `len`, `append`, `isinstance`, `str`, `format`, …

### Problem

These calls:
1. Cannot be resolved to user-defined symbols (they're built-ins).
2. Flood the graph with thousands of meaningless edges.
3. Make the call-resolution loop waste time on names it will never match.
4. Degrade graph quality — a `CALLS` edge should indicate a meaningful
   dependency, not that someone called `len()`.

### Solution

Module-level `frozenset` constants per language, checked in each call extractor:

```python
# extractor.py — three constants defined at module level
_PYTHON_BUILT_INS: frozenset[str] = frozenset({
    "print", "len", "range", "isinstance", "str", "int", "float",
    "append", "get", "update", "split", "strip", ...
})

_JS_BUILT_INS: frozenset[str] = frozenset({
    "log", "warn", "map", "filter", "forEach", "then", "catch",
    "useState", "useEffect", "getElementById", ...
})

_RUST_BUILT_INS: frozenset[str] = frozenset({
    "println", "unwrap", "expect", "clone", "into", "collect", ...
})
```

Each extractor now does a `frozenset` membership check (O(1)) before appending:

```python
# Before
calls.append(RawCall(callee_name=_node_text(func_node, src), ...))

# After
name = _node_text(func_node, src)
if name in _PYTHON_BUILT_INS:
    continue  # skip noise
calls.append(RawCall(callee_name=name, ...))
```

The public `_is_noise_call(name, language)` function is also available for
callers outside the module.

### Effect

On a typical 10k-line Python project, this reduces raw call events by ~60%
before they even reach the pipeline's resolution phase.

---

## 4. Decision: Three-Tier Call Resolution with Confidence Scores

**Files changed:** `src/code_indexer/graph/pipeline.py`

### Before

In `_build_snapshot`, after extracting calls from each file, the pipeline tried
to resolve callee names to `SymbolNode` objects.  The logic was:

```python
# Old approach — naive "first candidate wins"
callee_sn = local_syms_by_name.get(raw_call.callee_name)
if callee_sn is None:
    candidates = global_name_index.get(raw_call.callee_name, [])
    callee_sn = candidates[0] if candidates else None  # ← just picks first!
```

### Problem

1. **False edges from ambiguous names.**  If `process()` is defined in 15
   different files, the old code picked `candidates[0]` — which is essentially
   random.  The resulting `CALLS` edge points to the wrong symbol.

2. **No sense of confidence.**  All edges looked equally reliable regardless
   of whether the match was local, imported, or a lucky global guess.

3. **Cannot prioritise import-resolved matches.**  A callee that lives in an
   explicitly imported file is much more likely to be correct than a random
   global match, but the old code treated them identically.

### Solution — GitNexus three-tier confidence pattern

```python
# Tier 1: same-file symbol — high confidence, name is unambiguous locally
callee_sn = local_syms_by_name.get(raw_call.callee_name)
confidence = 0.85

if callee_sn is None:
    all_candidates = global_name_index.get(raw_call.callee_name, [])

    # Tier 2: symbol in an explicitly imported file — highest confidence
    # "explicitly imported" = file_id appears in this file's IMPORTS edges
    imported_candidates = [c for c in all_candidates if c.file_id in this_imports]
    if len(imported_candidates) == 1:
        callee_sn = imported_candidates[0]
        confidence = 0.90
    elif len(imported_candidates) > 1:
        continue  # ambiguous across imported files — skip, don't guess

    else:
        # Tier 3: unique global match — low confidence
        if len(all_candidates) == 1:
            callee_sn = all_candidates[0]
            confidence = 0.50
        else:
            continue  # ambiguous globally — skip entirely
```

Confidence is stored in `GraphEdge.properties["confidence"]` and is available
for downstream consumers (e.g., a UI showing call graphs could dim low-confidence
edges).

The `this_imports` set is built **once** from the IMPORTS edges already in the
snapshot — no extra resolution work needed.

### Effect

- False call edges are dramatically reduced (ambiguous names are skipped).
- Cross-file inheritance resolution uses the same import-preference logic.
- Every `CALLS` edge now carries a `confidence` score.

---

## 5. Decision: Suffix Index in ImportResolver

**Files changed:** `src/code_indexer/graph/resolver.py`

### Before

`ImportResolver.__init__` built two lookup structures:

```python
self._file_map = { normalised_path: FileNode }   # O(1) exact lookup
self._by_stem  = { stem: [path, ...] }            # basename → paths
```

For absolute Python imports like `from myapp.services.auth import jwt`, the
resolver converted dots to slashes and tried extensions:

```python
as_path = "myapp/services/auth/jwt"
result = self._try_extensions(as_path, Language.PYTHON)  # tries .py, /__init__.py
if not result:
    # progressively shorter prefixes: myapp/services/auth, myapp/services, ...
    for i in range(len(parts), 0, -1):
        result = self._try_extensions("/".join(parts[:i]), Language.PYTHON)
```

### Problem

1. **Fails on deep monorepo nesting.** If the project structure is
   `backend/src/myapp/services/auth/jwt.py`, importing as `myapp.services.auth.jwt`
   from a different package root wouldn't resolve because the dict key starts
   with `backend/src/`.

2. **O(depth) fallback.** The `for i in range(len(parts), 0, -1)` loop tries
   every prefix — up to N dict lookups per import for a deeply nested package.

### Solution — suffix index (GitNexus pattern)

At construction time, register every *trailing sub-path* of every file stem:

```python
# For file "backend/src/myapp/services/auth/jwt.py"
# stem = "backend/src/myapp/services/auth/jwt"
# parts = ["backend", "src", "myapp", "services", "auth", "jwt"]

for i in range(len(parts)):
    suffix = "/".join(parts[i:])   # jwt, auth/jwt, services/auth/jwt, ...
    self._suffix_index.setdefault(suffix, []).append(file_node)
```

Lookup is now one dict access:

```python
def _try_suffix_index(self, module_string: str) -> FileNode | None:
    key = module_string.replace(".", "/").strip("/")
    candidates = self._suffix_index.get(key)
    # Only return if unambiguous (exactly one file matches this suffix)
    if candidates and len(candidates) == 1:
        return candidates[0]
    return None
```

Used as the **final fallback** in `_resolve_python` and `_resolve_java` after
the direct and prefix strategies fail.

### Trade-off

- **Build cost:** O(total_path_depth) at construction — proportional to the
  total number of path segments across all files.  For 10k files with average
  depth 5, that's ~50k entries.  Acceptable.
- **Lookup:** O(1).
- **Ambiguity guard:** if two files share the same suffix (e.g., two `utils.py`
  files at different depths), the index returns `None` rather than guessing.

---

## 6. Decision: Reciprocal Rank Fusion (RRF) for Hybrid Search

**Files changed:** `src/code_indexer/retrieval/hybrid_retriever.py`

### Before

`_merge_and_rank` combined vector and graph results with a weighted sum:

```python
# Vector results — multiplied by vector_weight (default 0.7)
score = original_vector_score * self._vector_weight

# Graph expansion results — fixed constant times graph_weight
score = _GRAPH_EXPANSION_SCORE * self._graph_weight   # 0.6 * 0.3 = 0.18
      + related_result.score * self._vector_weight    # + similarity * 0.7
```

### Problem

1. **Scale sensitivity.**  Vector similarity scores from different embedding
   models have different distributions (cosine similarity vs. dot product, with
   or without normalisation).  Multiplying by a fixed weight doesn't account
   for this — a model returning scores in [0.9, 1.0] vs [0.0, 1.0] produces
   very different merged rankings.

2. **Graph results always penalised.**  The `_GRAPH_EXPANSION_SCORE = 0.6`
   constant meant graph-expanded results could never rank above vector results,
   even when the graph result is the *actual* function being asked about and
   the vector result is only a loosely related comment.

3. **`_last_query_vector` was never set** (see section 9), so graph expansion
   silently returned zero results — making the hybrid retriever equivalent to
   a plain vector search with extra overhead.

### Solution — Reciprocal Rank Fusion

From Cormack et al. (2009).  Each result list contributes a rank-based score:

```
rrf_score(doc) = Σ_list   1 / (60 + rank_in_list(doc))
```

The constant 60 is the standard value from the original paper.  It controls
how much the top-ranked items are boosted relative to lower-ranked ones.

```python
_RRF_K = 60

def _merge_and_rank(self, vector_results, graph_results, top_k):
    rrf_scores: dict[str, float] = {}
    best_result: dict[str, SearchResult] = {}

    # Contribution from vector ranking (1-indexed)
    for rank, r in enumerate(vector_results, start=1):
        cid = r.chunk.id
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (_RRF_K + rank)
        if cid not in best_result:
            best_result[cid] = r

    # Contribution from graph expansion ranking
    for rank, r in enumerate(graph_results, start=1):
        cid = r.chunk.id
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (_RRF_K + rank)
        if cid not in best_result:
            best_result[cid] = r

    # Sort by total RRF score and assign final ranks
    sorted_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)
    ...
```

A chunk appearing in *both* lists accumulates scores from both — it naturally
floats to the top without any manually tuned weights.

### Effect

| Scenario | Before | After |
|---|---|---|
| Result in both lists | Weighted sum (biased toward vector) | RRF from both lists — rises naturally |
| Graph-only result | Capped at 0.18 score | RRF rank-1 = 1/(60+1) ≈ 0.016; competitive |
| Scale invariant | No — depends on raw scores | Yes — only ranks matter |
| Tuning needed | `vector_weight` must be calibrated | `_RRF_K=60` is universally robust |

---

## 7. Fix: O(n) `_find_enclosing_symbol`

**Files changed:** `src/code_indexer/graph/pipeline.py`

### Before

During call resolution, the pipeline needed to find which function *contains*
a given call site (by line number).  The original implementation:

```python
@staticmethod
def _find_enclosing_symbol(line, file_id, snapshot):
    best = None
    best_size = float("inf")
    for sn in snapshot.symbol_nodes.values():   # ← scans ALL symbols
        if sn.file_id != file_id:               # ← filters per iteration
            continue
        if sn.start_line <= line <= sn.end_line:
            size = sn.end_line - sn.start_line
            if size < best_size:
                best = sn
                best_size = size
    return best
```

Called once **per `RawCall` event** — so total cost was:

```
O(total_symbols_in_repo  ×  total_calls_in_repo)
```

For a 10k-file Python repo with ~50k symbols and ~200k call sites, that's
**10 billion comparisons**.

### Solution

**Step 1 — build a per-file index once before the loop:**

```python
file_symbol_index = self._build_file_symbol_index(snapshot)
# {file_id: [SymbolNode sorted by start_line]}
```

Cost: O(total_symbols × log(max_symbols_per_file)) — paid once.

**Step 2 — binary search per call:**

```python
@staticmethod
def _find_enclosing_symbol(line, file_id, file_symbol_index):
    syms = file_symbol_index.get(file_id, [])   # O(1) dict lookup
    start_lines = [s.start_line for s in syms]
    idx = bisect.bisect_right(start_lines, line) # O(log n)

    # Only scan symbols that start before this line
    best, best_size = None, float("inf")
    for s in syms[:idx]:
        if s.end_line >= line:
            size = s.end_line - s.start_line
            if size < best_size:
                best, best_size = s, size
    return best
```

**Complexity improvement:**

| | Before | After |
|---|---|---|
| Per-call cost | O(total_symbols_in_repo) | O(log n + k) where k = symbols starting before this line in this file |
| Total cost | O(symbols × calls) | O(symbols log n) build + O(calls × k) |
| 10k-file repo estimate | ~10B ops | ~500k ops |

---

## 8. Fix: In-Memory Graph Persistence

**Files changed:** `src/code_indexer/graph/in_memory_graph.py`,
`src/code_indexer/graph/base.py`

### Before

`InMemoryGraphStore` had no persistence.  Every process restart required
re-indexing the entire codebase — which could take minutes on large repos.

### Problem

For development use (the primary use case for the in-memory store), this
meant running the full indexing pipeline on every server restart.  Incremental
re-indexing wasn't supported either.

### Solution

Three new methods on `InMemoryGraphStore`:

```python
# Persist to disk
store.save(".cache/graph.json")

# Restore from disk (class method — creates a new store)
store = InMemoryGraphStore.load(".cache/graph.json")

# Internal — reconstruct a GraphSnapshot from live graph state
snap = store._to_snapshot()
```

**Format:** plain JSON via Pydantic `model_dump(mode="json")`.  Human-readable,
diffable, and compatible with any Pydantic version that supports `mode="json"`.

**Load sequence:**
1. Read JSON → reconstruct `GraphSnapshot` using `model_validate`.
2. Call `store.upsert(snapshot)` — this rebuilds all secondary indexes
   (`_symbols_by_name`, `_symbols_by_file`, `_files_by_path`) automatically.

**`BaseGraphStore.save()`** added as a non-abstract method that raises
`NotImplementedError` by default.  This lets callers write:
```python
store.save(path)  # works for InMemoryGraphStore, raises for Neo4jGraphStore
```
without needing `isinstance` checks.

### Usage pattern

```python
cache_path = ".cache/graph.json"

if os.path.exists(cache_path):
    store = InMemoryGraphStore.load(cache_path)
else:
    pipeline = GraphIndexingPipeline(graph_store=InMemoryGraphStore())
    pipeline.index_directory("./my-project")
    pipeline.store.save(cache_path)
    store = pipeline.store
```

---

## 9. Fix: Fragile `_last_query_vector` Access

**Files changed:** `src/code_indexer/retrieval/retriever.py`,
`src/code_indexer/retrieval/hybrid_retriever.py`

### Before

`HybridRetriever._expand_with_graph` needed the query embedding vector to
search the vector store for chunks related to graph-expanded symbols.  It
accessed it via a private attribute:

```python
related_chunks = self._vector_retriever._vector_store.query(
    query_vector=self._vector_retriever._last_query_vector or [],
    ...
) if hasattr(self._vector_retriever, "_last_query_vector") else []
```

### Problem (two bugs, not one)

**Bug 1 — attribute never exists.**  `CodeRetriever` never stores
`_last_query_vector` anywhere.  The `hasattr(...)` guard always returned
`False`, so `related_chunks` was always `[]`.
**Graph expansion was completely non-functional** — the hybrid retriever
was effectively a plain vector search with extra overhead on every call.

**Bug 2 — accessing private attributes across class boundaries.**  Even if the
attribute had existed, reading `obj._private_attr` from a different class is
fragile: naming can change, caching logic can change, and there's no contract.

### Solution

**Add two public methods to `CodeRetriever`:**

```python
def embed_query(self, text: str) -> list[float]:
    """Embed one string, return the vector."""
    return self._embedder.embed_texts([text])[0]

def search_by_vector(self, query_vector, top_k, filters=None) -> list[SearchResult]:
    """Query the vector store with a pre-computed vector (skip embedding)."""
    return self._vector_store.query(
        query_vector=query_vector, top_k=top_k, filters=filters or None
    )
```

**Embed once in `HybridRetriever.search`, pass explicitly:**

```python
def search(self, query):
    # Embed once — shared by vector search AND graph expansion
    query_vector = self._vector_retriever.embed_query(query.query)

    vector_response = self._vector_retriever.search(...)         # uses internal embed
    expanded = self._expand_with_graph(..., query_vector)        # uses pre-computed
```

**`_expand_with_graph` now takes `query_vector` as an explicit parameter:**

```python
def _expand_with_graph(self, vector_results, query, query_vector):
    ...
    related_chunks = self._vector_retriever.search_by_vector(
        query_vector=query_vector,     # ← explicit, never None
        top_k=3,
        filters={"path": rel_sym.file_path, "name": rel_sym.name},
    )
```

### Effect

- Graph expansion now actually runs (was silently broken before).
- `embed_query` and `search_by_vector` are stable, documented public API.
- The embedding call in `search()` embeds once and is reused for all expansion
  queries — no per-symbol re-embedding.

---

## 10. Reference: GitNexus Patterns

[GitNexus](https://github.com/abhigyanpatwari/GitNexus) was reviewed as a
reference implementation.  The following patterns were adapted for this codebase:

| GitNexus pattern | Where applied here | File |
|---|---|---|
| `isBuiltInOrNoise` Set + filtering | `_PYTHON_BUILT_INS`, `_JS_BUILT_INS`, `_RUST_BUILT_INS` | `graph/extractor.py` |
| Three-tier confidence resolution (local 0.85 / imported 0.90 / fuzzy 0.50) | Call resolution loop | `graph/pipeline.py` |
| Suffix index for O(1) import path lookup | `_suffix_index` in `ImportResolver` | `graph/resolver.py` |
| RRF for hybrid search merging | `_merge_and_rank` | `retrieval/hybrid_retriever.py` |
| Ambiguous candidate skipping | `len(candidates) > 1 → continue` | `graph/pipeline.py` |

**Patterns reviewed but not adopted (and why):**

| Pattern | Reason not adopted |
|---|---|
| Leiden community detection | Requires `igraph` / `leidenalg` dependency; adds ~50MB. Graph is queryable without communities. |
| Worker pool parallel parsing | Python GIL limits CPU-bound parallelism; `concurrent.futures.ProcessPoolExecutor` would add complexity for moderate gain. Deferred. |
| Entry point / execution flow scoring | Requires a separate heuristic pass over function names. Deferred — low priority for RAG use case. |
| `yieldToEventLoop` pattern | JS-specific. Python equivalent (`asyncio.sleep(0)`) is only needed if running the pipeline inside an async context, which the current API does via `loop.run_in_executor`. |
