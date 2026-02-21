# Impact Analysis

This document explains how the code-indexer derives impact analysis — answering
the question: **"If I change symbol X, what else could break?"**

---

## Overview

Impact analysis runs as a graph traversal over the code knowledge graph built
by `GraphIndexingPipeline`.  Three independent dimensions of impact are
combined into a single `ImpactResult`:

| Dimension | Graph edges traversed | Direction |
|-----------|----------------------|-----------|
| Call-graph | `CALLS` | Reverse (incoming) |
| Inheritance | `INHERITS_FROM` | Reverse (incoming) |
| File imports | `IMPORTS` | Reverse (incoming) |

---

## 1. Call-graph reverse reachability

### What it answers
Which functions/methods transitively call the changed symbol and could
therefore behave differently after the change?

### Approach

**Extraction (index time)**

Tree-sitter parses every source file and extracts raw `call_expression` nodes.
Each call produces a `RawCall(callee_name, callee_object, line)`.  Built-in
and overly-generic names (e.g. `len`, `map`, `get`) are filtered out before
they reach the graph.

**Resolution (index time)**

Raw calls are resolved to concrete `SymbolNode` targets using a three-tier
name-match strategy:

| Tier | Candidate set | Confidence |
|------|--------------|-----------|
| 1 | Symbol defined in the **same file** | 0.85 |
| 2 | Symbol defined in an **explicitly imported file** (unique match) | 0.90 |
| 3 | **Unique global match** across the entire codebase | 0.50 |

Ambiguous matches (multiple candidates at any tier) are **dropped** to avoid
false edges.  Each resolved `CALLS` edge stores its confidence in
`edge.properties["confidence"]`.

**Traversal (query time)**

`InMemoryGraphStore.get_impact_set` runs a BFS over reversed `CALLS` edges
starting from the changed symbol.  At each hop, the **compound confidence** is
updated as:

```
compound = min(path_confidence_so_far, edge.confidence)
```

Using `min` rather than multiplication is deliberately conservative: a single
uncertain link in the chain caps the whole path.  Edges whose compound
confidence falls below `min_confidence` are pruned.

The result separates callers into:

* **Certain** (compound ≥ 0.85) — same-file or uniquely-imported calls.
* **Probable** (0.50–0.84) — globally-unique matches or mixed-confidence chains.
* **Speculative** (< 0.50) — low-confidence chains; useful for discovery,
  should not drive blocking decisions.

### Limitations

Tree-sitter provides syntactic call extraction only — no type inference.  This
means:

* **Polymorphism** is not resolved.  If two classes both define `process()`,
  a call to `obj.process()` where the type of `obj` is unknown will produce
  two candidates and will be **dropped** (ambiguous, tier-3 rule).
* **Dynamic dispatch** (e.g. `getattr`, function pointers, decorators) is not
  tracked.
* **False negatives** are possible at tier 3 when a callee name is common
  enough to collide globally.

For higher precision, a language-server backend (pyright, rust-analyzer,
typescript-language-server) can be wired in to replace or augment tier-3
resolution.

---

## 2. Inheritance reverse (subclasses)

### What it answers
Which classes or interfaces inherit from the changed symbol and may need to be
updated when its API changes?

### Approach

**Extraction**

Each language extractor identifies `class_definition` / `interface_declaration`
nodes with a base-class list.  The result is a `RawInheritance(class_name,
base_names)`.

**Resolution**

The pipeline resolves base names using the same two-pass strategy as calls:
intra-file first, then cross-file via import edges.  Each resolved pair becomes
an `INHERITS_FROM` edge.

**Traversal**

`get_impact_set` calls `get_subclasses(symbol_id)` — a single hop over
reversed `INHERITS_FROM` edges.  Only direct subclasses are returned; transitive
subclass hierarchies require multiple calls or a dedicated deep-inheritance query.

---

## 3. File-level reverse imports

### What it answers
Which files directly depend on the file containing the changed symbol?  These
files always need human review regardless of call-graph confidence, because any
re-export, module-level constant, or decorator from the changed file could
affect them.

### Approach

**Extraction**

Import statements are extracted exactly from the syntax tree (no name matching
needed — import strings are literals).  Relative imports are resolved against
the indexing root; external (third-party) imports are recorded but not linked.

**Resolution**

`ImportResolver` maps each import string to a concrete `FileNode` in the
repository.  The result is an `IMPORTS` edge from the importing file to the
imported file.

**Traversal**

`get_impact_set` returns all files that have a direct `IMPORTS` edge pointing
to the changed symbol's file.  Import resolution is exact, so this dimension
has no confidence issue — all entries are certain.

---

## API

### `POST /graph/impact`

```json
{
  "symbol_id": "<stable-id>",
  "min_confidence": 0.85
}
```

Or by name:

```json
{
  "symbol_name": "verify_token",
  "min_confidence": 0.0
}
```

**Response: `ImpactResult`**

```json
{
  "symbol": { ... },
  "direct_callers":      [ ... ],
  "transitive_callers":  [ ... ],
  "subclasses":          [ ... ],
  "importing_files":     [ ... ],
  "affected_files":      [ ... ],
  "confidence_breakdown": {
    "certain":     12,
    "probable":     3,
    "speculative":  1
  },
  "min_confidence_used": 0.0
}
```

**Fields**

| Field | Description |
|-------|-------------|
| `direct_callers` | Symbols with a direct `CALLS` edge (1 hop). |
| `transitive_callers` | All reachable callers (unlimited BFS). Includes direct callers. |
| `subclasses` | Direct subclasses / implementors. |
| `importing_files` | Files that directly import the changed symbol's file. |
| `affected_files` | Deduplicated files containing any transitive caller or subclass. Use this for retest planning. |
| `confidence_breakdown` | Caller counts by tier. |
| `min_confidence_used` | The threshold applied to the call-graph traversal. |

**`min_confidence` guidance**

| Value | Use case |
|-------|----------|
| `0.0` | Full speculative set — maximum recall, some false positives. |
| `0.50` | Probable + certain — good default for most impact analysis. |
| `0.85` | Certain only — use when you need a tight, low-noise impact set. |

---

## Programmatic usage

```python
from code_indexer.graph.pipeline import GraphIndexingPipeline, build_graph_store
from code_indexer.core.config import get_settings

settings = get_settings()
pipeline = GraphIndexingPipeline(graph_store=build_graph_store(settings))
pipeline.index_directory("./my-project")

# Find the symbol
store = pipeline.store
[sym] = store.find_symbols_by_name("verify_token", kind="function")

# Run impact analysis (certain + probable callers only)
impact = store.get_impact_set(sym.id, min_confidence=0.50)

print(f"Direct callers:     {len(impact.direct_callers)}")
print(f"Transitive callers: {len(impact.transitive_callers)}")
print(f"Affected files:     {len(impact.affected_files)}")
print(f"Confidence:         {impact.confidence_breakdown}")

for f in impact.affected_files:
    print(f"  retesting: {f.path}")
```

---

## Accuracy roadmap

The current implementation is a pragmatic balance of speed (tree-sitter, no
language server) and correctness (confidence-weighted BFS, ambiguity pruning).
For higher precision:

1. **Language-server integration** — wire pyright / rust-analyzer to replace
   tier-3 global name matching with type-resolved targets.
2. **Deep inheritance traversal** — extend `get_impact_set` to follow
   `INHERITS_FROM` edges transitively (multi-hop subclass chains).
3. **Transitive reverse imports** — BFS over reversed `IMPORTS` edges to
   surface files that are indirectly affected through import chains.
4. **Signature-aware impact** — track which parameters/return types changed
   to filter out callers that only use unaffected parts of the API.
