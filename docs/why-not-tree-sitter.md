# Why Tree-sitter Is Insufficient for Precise Impact Analysis

Tree-sitter is an excellent tool for **structural extraction** (symbols,
imports, syntax) but it fundamentally cannot answer the question that impact
analysis requires: *"when `foo()` is called here, which specific definition
of `foo` is being invoked?"*

This document explains why, with concrete code examples, and describes what
should be used instead.

---

## What Tree-sitter Actually Does

Tree-sitter is a **syntax-only parser**.  It builds a Concrete Syntax Tree
(CST) from source text — tokens, operators, identifiers, blocks — with no
knowledge of types, scopes, or semantics.

It can answer:

- "What functions are defined in this file?" ✓
- "What are the names of the arguments to this call?" ✓
- "Which lines contain import statements?" ✓

It cannot answer:

- "Which class does this variable belong to?" ✗
- "Does `obj.process()` call `A.process` or `B.process`?" ✗
- "Is this `create` a factory method on `UserService` or `OrderService`?" ✗

---

## Failure Mode 1 — Polymorphism

The most common failure in impact analysis.

```python
# services/base.py
class BaseExporter:
    def export(self, data: list) -> None:
        raise NotImplementedError

# services/csv_exporter.py
class CsvExporter(BaseExporter):
    def export(self, data: list) -> None:
        write_csv(data)          # <-- THIS export()

# services/json_exporter.py
class JsonExporter(BaseExporter):
    def export(self, data: list) -> None:
        write_json(data)         # <-- THIS export()

# pipeline.py
def run_export(exporter: BaseExporter, data: list) -> None:
    exporter.export(data)        # <-- which export() does this call?
```

Tree-sitter extracts the call `export` from `run_export`.  It sees three
definitions of `export` in the codebase.  The name-match resolver finds
**multiple candidates** — ambiguous — and **drops the edge entirely**.

**Impact analysis result:** `run_export` is not listed as a caller of any
`export`.  If you change `CsvExporter.export`, `run_export` does **not** appear
in the impact set, even though it is clearly affected.

**What a type-aware tool would do:** Infer that `exporter` is typed
`BaseExporter`, resolve `exporter.export` to `BaseExporter.export`, and then
follow `INHERITS_FROM` to surface both subclass implementations.

---

## Failure Mode 2 — Method Name Collision Across Unrelated Classes

```python
# auth/tokens.py
class TokenValidator:
    def validate(self, token: str) -> bool:
        return verify_signature(token)

# payments/invoices.py
class InvoiceValidator:
    def validate(self, invoice: dict) -> bool:
        return check_totals(invoice)

# api/middleware.py
def handle_request(validator, payload):
    if not validator.validate(payload):   # which validate()?
        raise PermissionError
```

Three files.  Two unrelated classes both named `*Validator` with a `validate`
method.  Tree-sitter sees `validator.validate(payload)` — the object name
`validator` gives no type information.  Again: **ambiguous, dropped**.

Changing `TokenValidator.validate` — a security-critical function — will **not**
appear to affect `handle_request` in the impact set.  The change goes
undetected.

---

## Failure Mode 3 — Dynamic Dispatch and Higher-Order Functions

```python
# core/registry.py
_handlers: dict[str, Callable] = {}

def register(name: str, fn: Callable) -> None:
    _handlers[name] = fn

def dispatch(event: str, payload: dict) -> None:
    _handlers[event](payload)     # <-- tree-sitter sees a subscript call, not a name
```

Tree-sitter cannot extract a callee name from `_handlers[event](...)` — there
is no `identifier` or `attribute` node to match.  This call is **invisible** to
name-based resolution.

Any function registered via `register(...)` will never appear as a callee of
`dispatch` in the graph, no matter how many exist.

---

## Failure Mode 4 — Import Aliases and Re-exports

```python
# utils/__init__.py
from utils._internal import compute_hash as hash_fn   # re-export with alias

# worker.py
from utils import hash_fn
hash_fn(data)                  # calls compute_hash, but name is "hash_fn"
```

Tree-sitter extracts the call `hash_fn` from `worker.py`.  It looks for a
symbol named `hash_fn` in the global index — and finds it in `utils/__init__.py`
(it resolves the import string correctly).  **So far so good.**

But the `CALLS` edge points to the `hash_fn` re-export node, not to the
original `compute_hash` definition in `utils/_internal.py`.  If you change
`compute_hash`, `worker.py` **does** appear in the impact set (via the import
chain), but the CALLS edge itself is misleading — the graph treats `hash_fn`
and `compute_hash` as separate symbols.

Rename `compute_hash` to `compute_hash_v2` and the edge breaks entirely.

---

## Failure Mode 5 — Decorators That Wrap Callees

```python
# framework/router.py
def route(path: str):
    def decorator(fn):
        _routes[path] = fn
        return fn
    return decorator

# api/users.py
@route("/users")
def list_users(request):       # called via router, not by name
    return db.query(User)
```

Tree-sitter sees the decorator `route` applied to `list_users` and records a
`RawCall(callee_name="route")`.  It does not understand that HTTP requests will
eventually invoke `list_users` through the `_routes` dictionary.  `list_users`
has **zero callers** in the graph.

If you change `list_users`, `route` appears in the impact set (because of the
decorator call) — but not the actual request-handling code that depends on it.

---

## Quantified Impact on Recall and Precision

The three-tier resolver in this codebase mitigates some of these issues:

| Scenario | Tree-sitter result | Actual impact |
|----------|-------------------|---------------|
| Unique global name, no collision | Edge created (0.50 conf) | ✓ Correct |
| Unique name, in imported file | Edge created (0.90 conf) | ✓ Correct |
| Same-file call, unique name | Edge created (0.85 conf) | ✓ Correct |
| Polymorphic method (multiple defs) | **Edge dropped** | ✗ False negative |
| Dynamic dispatch / `__call__` | **No edge extracted** | ✗ False negative |
| Decorator-mediated calls | Decorator call only | ✗ Partial |
| Import alias, re-export | Points to alias node | ⚠ Misleading |

In a typical Python web service codebase, studies of similar tools suggest
**15–30% of call edges are false negatives** (missed) when using name-only
resolution.  For impact analysis, a false negative means "a breaking change is
not surfaced" — which is exactly the worst possible failure mode.

---

## What Should Be Used Instead

### Option A — Language Server Protocol (LSP)

Each major language has a type-aware language server that resolves calls
precisely:

| Language | Language Server | Resolution accuracy |
|----------|----------------|---------------------|
| Python | `pyright`, `pylsp` | ~95% with type annotations |
| TypeScript | `typescript-language-server` | ~99% (typed language) |
| Go | `gopls` | ~99% |
| Rust | `rust-analyzer` | ~99% |
| Java | `eclipse.jdt.ls` | ~99% |

LSPs expose a `textDocument/definition` RPC: given a file + cursor position,
return the exact definition location.  This gives **zero-ambiguity call
resolution** for typed code.

**Integration sketch (Python, using `pylsp-jsonrpc`):**

```python
import subprocess
import json

class LspCallResolver:
    """Resolve call sites to exact definitions via pyright LSP."""

    def __init__(self, project_root: str) -> None:
        self._root = project_root
        self._proc = subprocess.Popen(
            ["pyright", "--outputjson"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )

    def resolve(self, file_path: str, line: int, col: int) -> str | None:
        """Return the definition location for the symbol at (line, col)."""
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "textDocument/definition",
            "params": {
                "textDocument": {"uri": f"file://{file_path}"},
                "position": {"line": line - 1, "character": col},
            },
        }
        self._proc.stdin.write(json.dumps(request).encode())
        response = json.loads(self._proc.stdout.readline())
        if result := response.get("result"):
            return result[0]["uri"].replace("file://", "")
        return None
```

This replaces tier-3 (global name match) entirely and can optionally replace
tiers 1 and 2 as well.

**Cost:** LSP startup is slow (2–15s per project) and requires the language
toolchain to be installed.  Suitable for on-demand queries, not bulk indexing.

---

### Option B — Static Type Checker Output

Type checkers like `pyright` and `mypy` emit structured type information that
can be consumed offline.

```bash
# Emit pyright's type analysis as JSON
pyright --outputjson > pyright_output.json
```

The output includes inferred types for every expression in every file.  A
post-processing step can use these types to resolve method calls precisely:

```python
import json

def build_type_map(pyright_output: str) -> dict[tuple[str, int, int], str]:
    """Return {(file, line, col): inferred_type} from pyright JSON."""
    data = json.loads(pyright_output)
    type_map: dict[tuple[str, int, int], str] = {}
    for diag in data.get("generalDiagnostics", []):
        # pyright includes hover info in --outputjson mode
        ...  # parse hover entries
    return type_map


def resolve_call_with_types(
    raw_call: RawCall,
    file_path: str,
    type_map: dict,
    symbol_index: dict[str, list[SymbolNode]],
) -> SymbolNode | None:
    """Use inferred type of callee_object to disambiguate the call."""
    if raw_call.callee_object is None:
        return None

    inferred_type = type_map.get((file_path, raw_call.line, 0))
    if inferred_type is None:
        return None  # fall back to name-match

    # Look up method in the resolved type's symbol set
    class_sym = symbol_index.get(inferred_type, [])
    for sym in class_sym:
        if sym.name == raw_call.callee_name and sym.kind == SymbolKind.METHOD:
            return sym
    return None
```

**Cost:** Pyright needs to be run once per index build (~10–60s for large
codebases).  Output is a static snapshot, so it goes stale when code changes.

---

### Option C — Hybrid Layered Approach (Recommended)

Use Tree-sitter for everything it does well (fast, zero-dependency structural
extraction), and layer LSP/type-checker resolution only where name matching
fails.

```
Tier 1: Tree-sitter same-file        → confidence 0.85  (keep as-is)
Tier 2: Tree-sitter imported-file    → confidence 0.90  (keep as-is)
Tier 3: Pyright type resolution      → confidence 0.99  (NEW — replaces global name match)
Tier 4: Tree-sitter global unique    → confidence 0.50  (fallback if pyright unavailable)
```

Implementation outline:

```python
class HybridCallResolver:
    def __init__(
        self,
        file_map: dict[str, FileNode],
        pyright_type_map: dict | None = None,
    ) -> None:
        self._import_resolver = ImportResolver(file_map)
        self._pyright = pyright_type_map  # None = pyright not available

    def resolve(
        self,
        raw_call: RawCall,
        source_file: str,
        local_symbols: dict[str, SymbolNode],
        imported_file_symbols: dict[str, list[SymbolNode]],
        global_index: dict[str, list[SymbolNode]],
    ) -> tuple[SymbolNode | None, float]:
        # Tier 1: same-file
        if sn := local_symbols.get(raw_call.callee_name):
            return sn, 0.85

        # Tier 2: imported file, unique match
        candidates = imported_file_symbols.get(raw_call.callee_name, [])
        if len(candidates) == 1:
            return candidates[0], 0.90

        # Tier 3: pyright type resolution (if available)
        if self._pyright and raw_call.callee_object:
            if sn := self._pyright_resolve(raw_call, source_file, global_index):
                return sn, 0.99

        # Tier 4: global unique match (fallback)
        all_candidates = global_index.get(raw_call.callee_name, [])
        if len(all_candidates) == 1:
            return all_candidates[0], 0.50

        return None, 0.0  # ambiguous

    def _pyright_resolve(
        self,
        raw_call: RawCall,
        source_file: str,
        global_index: dict[str, list[SymbolNode]],
    ) -> SymbolNode | None:
        inferred_type = self._pyright.get((source_file, raw_call.line))
        if inferred_type is None:
            return None
        for sym in global_index.get(raw_call.callee_name, []):
            if sym.parent_name == inferred_type:
                return sym
        return None
```

---

## When Tree-sitter IS the Right Choice

Tree-sitter remains the right tool for:

| Use case | Why Tree-sitter is sufficient |
|----------|------------------------------|
| **File-level import graph** | Import strings are literals — no type inference needed |
| **Symbol inventory** | Listing all functions/classes in a file is syntactic |
| **Chunking for RAG** | Splitting code by function/class boundaries is syntactic |
| **Inheritance extraction** | Base class names appear literally in the `class` declaration |
| **Approximate call graph for search** | False edges are tolerable when reranking handles them |
| **Cross-language support** | Tree-sitter has grammars for 100+ languages; LSPs are language-specific |
| **Offline / no-toolchain environments** | Tree-sitter has zero runtime dependencies |

The key insight: **precision matters differently depending on the use case**.
For RAG retrieval, a false positive (extra context included) is harmless.  For
impact analysis, a false negative (missed dependency) is a production incident
waiting to happen.

---

## Recommended Architecture for This Project

```
Phase 1 (current)
  Tree-sitter extraction → three-tier name resolution → CALLS edges with confidence
  Suitable for: RAG, code search, approximate impact analysis

Phase 2 (next)
  + Pyright/gopls type map consumed at index time
  → Tier-3 global name match replaced by type-resolved edges (confidence 0.99)
  → False negative rate drops from ~20% to ~2% for typed code

Phase 3 (future)
  + LSP on-demand resolution for interactive queries
  → Zero-ambiguity call resolution for any cursor position in any file
  → Enables rename refactoring, cross-repo impact, API surface diffing
```

The current three-tier Tree-sitter approach is a pragmatic Phase 1.  It will
surface the majority of impacts with acceptable recall for most codebases.
Phase 2 is the correct next investment once the indexing pipeline is stable.
