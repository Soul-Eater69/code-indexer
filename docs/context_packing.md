# Context Packing — Knapsack Selection for LLM Prompts

This document explains how `ContextPacker` and `GraphSnapshot.compute_katz_centrality()`
work together to fill a token-budget context window with the highest-value
code symbols.

---

## The Problem

A typical RAG pipeline retrieves a ranked list of candidate symbols — via
vector similarity, keyword search, or graph traversal — and concatenates
them into an LLM prompt.  This naive approach has two failure modes:

| Failure mode | Cause | Effect |
|---|---|---|
| **Budget overflow** | Total token cost > context window | Truncation cuts the most relevant (often largest) symbols |
| **Information flooding** | Too many low-relevance candidates | "Lost in the middle" effect; model ignores most context |

The fix is to *pack* the context: select the highest-value subset that fits
exactly within the budget.  This is the classic **0/1 knapsack problem**.

> Inspiration: a comment in the r/LocalLLaMA *code-chopper* thread (2025):
> *"knapsack optimization so you only load the optimal context for the agent."*

---

## Katz Centrality — Scoring Structural Importance

Before packing, we need a relevance score for each symbol.  Two sources
combine naturally:

* **Vector similarity** — semantic closeness to the query.
* **Structural importance** — how architecturally central the symbol is.

`GraphSnapshot.compute_katz_centrality()` computes Katz centrality on the
call / injection / inheritance graph to quantify structural importance.
Inspired by the `entity_rank` use-case in `code-chopper`.

### What it measures

Katz centrality assigns higher scores to symbols that many others
transitively depend on — the architectural hotspots.

```
k_v = α × Σ_u A_uv k_u + β
```

Where `A_uv = 1` if symbol `u` depends on symbol `v` via a `CALLS`,
`INJECTS`, or `INHERITS_FROM` edge.  The iterative power-method converges
in O(max_iter × |E|) time with no external dependencies.

### Example

```python
from code_indexer.graph.pipeline import GraphIndexingPipeline, build_graph_store
from code_indexer.core.config import get_settings

pipeline = GraphIndexingPipeline(graph_store=build_graph_store(get_settings()))
pipeline.index_directory("./my-project")
snapshot = pipeline._last_snapshot  # or store the snapshot on index()

# Compute Katz centrality
katz = snapshot.compute_katz_centrality()

# Top-10 hotspots
top10 = sorted(katz.items(), key=lambda kv: kv[1], reverse=True)[:10]
for sym_id, score in top10:
    sym = snapshot.symbol_nodes[sym_id]
    print(f"  {score:.3f}  {sym.qualified_name}  [{sym.kind.value}]")
```

Example output:
```
  1.000  auth.jwt.JWTHandler.verify_token  [method]
  0.891  auth.jwt.JWTHandler  [class]
  0.743  core.db.Session.execute  [method]
  0.701  core.db.Session  [class]
  ...
```

### Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `alpha` | `0.1` | Attenuation factor.  Increase toward `0.3` for larger repos where multi-hop influence should propagate further.  Must be < `1/λ_max`. |
| `beta` | `1.0` | Baseline score for all nodes (prevents zero scores for isolated symbols). |
| `max_iter` | `100` | Power-method iterations.  Converges well before this for most codebases. |
| `tol` | `1e-6` | Stop early when total score change < `tol × n`. |
| `edge_types` | `{CALLS, INJECTS, INHERITS_FROM}` | Which edge types contribute to centrality.  Add `REFERENCES` for broader influence. |

---

## Context Packer

`ContextPacker.pack()` takes a list of `(SymbolNode, score)` pairs and a
token budget, and returns a `PackedContext` with the optimal subset.

### Token cost estimation

Token cost is estimated from the symbol's line span without source access:

```
token_cost(sym) = ⌈(end_line − start_line + 1) × tokens_per_line⌉
```

Default `tokens_per_line = 15.0`.  Tune per language:

| Language | Recommended `tokens_per_line` |
|----------|-------------------------------|
| Python / TypeScript | 15 (default) |
| Java (with Javadoc) | 25 |
| Rust / Go | 12 |
| C / C++ | 18 |

### Algorithms

**`greedy`** (default, O(n log n))

Sort by value density (`score / token_cost`) descending, then greedily
pick until budget is exhausted.  This is optimal for the fractional
knapsack and near-optimal (~95–99% of the DP solution) for the 0/1 variant
with typical relevance distributions.  Use this for production.

**`dp`** (exact, O(n × ⌈budget/quantum⌉))

Token costs are quantised to `token_quantum`-token buckets (default 64)
to cap the DP table.  At 8 192-token budget / 64-token quantum: 128 columns.
For n=200 symbols: 200 × 128 = 25 600 cells ≈ 200 KB.  Slower but
guarantees the globally optimal selection.  Use when the candidate set is
small (< 200) and the budget is tight.

### Full example

```python
from code_indexer.graph.context_packer import ContextPacker
from code_indexer.graph.models import GraphSnapshot

# --- 1. Score candidates -----------------------------------------------
# Mix of vector similarity (from your retriever) + Katz centrality
katz = snapshot.compute_katz_centrality()

candidates = [
    (sym, 0.6 * vector_score + 0.4 * katz.get(sym.id, 0.0))
    for sym, vector_score in vector_search_results   # list[(SymbolNode, float)]
]

# --- 2. Pack the context window ----------------------------------------
packer = ContextPacker(tokens_per_line=15.0)
packed = packer.pack(
    candidates,
    token_budget=8192,
    strategy="greedy",   # or "dp" for small candidate sets
    min_score=0.05,      # drop very low-relevance candidates first
)

print(packed.summary())
# PackedContext: 14 symbols, 7841/8192 tokens (96% utilization), 6 dropped

# --- 3. Use in a prompt ------------------------------------------------
print(packed.to_context_text())
# === Context window: 14 symbols (7841 est. tokens) ===
#   [method] auth.jwt.JWTHandler.verify_token  auth/jwt.py:42-89  (score=0.931)
#   [class]  auth.jwt.JWTHandler               auth/jwt.py:10-120 (score=0.871)
#   ...
```

### Combining with `SymbolContext.to_mermaid()`

For LLM prompts that need both structural topology and packed source context:

```python
ctx = store.get_symbol_context(focal_sym.id)
packed = packer.pack(candidates, token_budget=6000)

prompt = f"""
{ctx.to_mermaid()}

{packed.to_context_text()}

Question: {user_query}
"""
```

The Mermaid diagram gives the model a compact structural overview (a few
hundred tokens), and the packed context delivers the most relevant source
symbols within the remaining budget.

---

## `PackedContext` fields

| Field | Type | Description |
|-------|------|-------------|
| `symbols` | `list[SymbolNode]` | Selected symbols, ordered by descending score |
| `scores` | `list[float]` | Relevance score for each selected symbol |
| `total_tokens` | `int` | Estimated total token cost |
| `budget` | `int` | Token budget that was applied |
| `utilization` | `float` | `total_tokens / budget` (0–1) |
| `dropped` | `int` | Candidates excluded (too large or below `min_score`) |
| `strategy` | `str` | `"greedy"` or `"dp"` |

---

## Design notes

### Why Katz and not PageRank?

PageRank normalises by out-degree; in a code graph the denominator is
often 1 (functions typically call few others), making PageRank equivalent
to in-degree weighting.  Katz centrality avoids this by using a simple
additive attenuation and is parameter-free (just tune `alpha`).

### Why greedy over DP by default?

For a 8 192-token budget with 15 tokens/line, most symbols cost 100–500
tokens (7–33 lines).  The greedy heuristic achieves ≥ 95% of the DP
optimum in this regime while being orders of magnitude faster and requiring
no quantisation.  The DP option is provided for tight budgets (< 2 048
tokens) where the greedy gap can be more significant.

### Score weighting guidance

| Weight mix | Best for |
|---|---|
| `1.0 × vector, 0.0 × katz` | Pure semantic search (query is very specific) |
| `0.6 × vector, 0.4 × katz` | Balanced (default recommendation) |
| `0.3 × vector, 0.7 × katz` | Architectural exploration (query is broad) |
| `0.0 × vector, 1.0 × katz` | Hotspot analysis / repo overview |

---

## Roadmap

| Status | Item |
|--------|------|
| ✅ Done | `compute_katz_centrality()` on `GraphSnapshot` |
| ✅ Done | `ContextPacker` with greedy and DP strategies |
| ✅ Done | `PackedContext.to_context_text()` |
| Next | `tokens_per_line` auto-tuned per-language from `SymbolNode.language` |
| Next | Entrypoint-based traversal: expand from a symbol outward through edges until budget is hit, then pack |
| Future | Semantic deduplication: drop symbols whose content overlaps > 80% before packing |
| Future | Multi-file context: pack at the file level (whole files) for import-heavy queries |
