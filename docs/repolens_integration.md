# RepoLens Integration — Developer Reference

> **Audience:** Developers building or extending the indexer, or developers
> learning how graph-RAG for code actually works at implementation level.
> Assumes you understand Python, know roughly what RAG is, and are curious
> about the "why" behind each design choice.

---

## Table of Contents

1. [Why vanilla RAG breaks on code](#1-why-vanilla-rag-breaks-on-code)
2. [The vocabulary gap in detail](#2-the-vocabulary-gap-in-detail)
3. [split_identifier — implementation and rationale](#3-split_identifier--implementation-and-rationale)
4. [TermEnricher — offline knowledge extraction](#4-termenricher--offline-knowledge-extraction)
5. [TermChunk — term-centric functionality summarisation](#5-termchunk--term-centric-functionality-summarisation)
6. [ConcernClusterer — online concern grouping](#6-concernclusterer--online-concern-grouping)
7. [Concern blocks in the prompt](#7-concern-blocks-in-the-prompt)
8. [LLM client layer — swappable backends](#8-llm-client-layer--swappable-backends)
9. [Full data flow — end to end](#9-full-data-flow--end-to-end)
10. [Sources and prior work](#10-sources-and-prior-work)

---

## 1. Why vanilla RAG breaks on code

Standard RAG pipelines are designed for prose — documentation, articles,
support tickets.  The assumption is that users write queries in the same
vocabulary that appears in the documents.

Code violates this assumption in three specific ways:

### 1a. Vocabulary mismatch (abbreviations)

```python
def handle_req(req: HttpRequest, svc: AuthSvc, tx: DbTransaction):
    ...
```

A user querying *"how are HTTP requests authenticated?"* will produce an
embedding that is semantically close to tokens like `HTTP`, `request`,
`authenticate`.  The function above only contains `req`, `svc`, `tx` — short
abbreviations whose embeddings are far from the query in the vector space.

The function is missed even though it is the exact right answer.

### 1b. Concern tangling (long functions, multiple topics)

```python
def process_order(order_id: int, user: User, payment: PaymentInfo):
    # 1. validate the order exists   (lines 10-20)
    # 2. check user credit limit     (lines 21-35)
    # 3. apply discount codes        (lines 36-55)
    # 4. charge the payment method   (lines 56-90)
    # 5. send confirmation email     (lines 91-110)
```

With standard chunking this entire 110-line function becomes one embedding.
A query about *"discount code logic"* retrieves this function (correct), but
now the AI reads 90 lines of payment and email code to find 19 lines of
discount logic.  Signal-to-noise ratio is poor.

### 1c. Concern scattering (cross-file topics)

Related logic often lives in many files.  JWT validation might span:
- `auth/jwt.py` (token parsing and signing)
- `auth/middleware.py` (enforcement on routes)
- `api/deps.py` (FastAPI dependency injection)
- `tests/test_jwt.py` (tests)

A flat ranked list from a vector search mixes these with unrelated results.
There is no structure that says *"these 4 are all about the same topic."*

### The RepoLens approach

The features in sections 3–7 address each of these problems:

| Problem | Feature |
|---|---|
| Abbreviation mismatch | `split_identifier` + `TermEnricher` |
| Concern tangling | `TermChunkExtractor` |
| Concern scattering | `ConcernClusterer` |

---

## 2. The vocabulary gap in detail

The **vocabulary gap** is the term used in information retrieval research for
the mismatch between user query language and document language.  In academic
IR the fix is *query expansion* — expanding the query with synonyms and
related terms before searching.

For code the problem is one layer deeper: it is not just that synonyms are
missing, it is that the document language is systematically compressed via
a convention (abbreviation + camelCase) that is opaque to a general-purpose
embedding model.

The model has seen `req` thousands of times in its training data — as a
Python abbreviation, as a `requests` library shorthand, as a random word.
Its embedding for `req` is an average over all those contexts.  It is not
reliably close to `HTTP request`.

**Our fix:** expand the abbreviation to its full form in the specific
codebase context *before* embedding.  This is more reliable than query
expansion because it is grounded in the actual source code.

---

## 3. split_identifier — implementation and rationale

```
src/code_indexer/graph/term_utils.py
```

### What it does

Breaks any code identifier into a list of lowercase word tokens using a
single regex pass:

```python
from code_indexer.graph.term_utils import split_identifier

split_identifier("getUserByEmail")     # → ["get", "user", "by", "email"]
split_identifier("MAX_RETRY_COUNT")   # → ["max", "retry", "count"]
split_identifier("parseHTTPResponse") # → ["parse", "http", "response"]
split_identifier("req")               # → ["req"]
```

### The regex

The function uses a two-step approach:

1. **camelCase / PascalCase split** — insert a split boundary before any
   uppercase letter that is preceded by a lowercase letter, or before any
   uppercase letter that is followed by a lowercase letter (handles `HTTP`
   in `parseHTTPResponse`):

   ```python
   import re
   _CAMEL_RE = re.compile(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
   ```

2. **Snake case split** — split on underscores, drop empty strings.

3. **Lowercase + length filter** — lowercase everything, drop tokens shorter
   than 2 characters (removes single-letter noise like the `i` in `i_count`).

### Why not just use a tokeniser?

A BPE tokeniser (the kind used by LLMs) would split `getUserByEmail` into
subword tokens, but those subwords are chosen to minimise vocabulary size,
not to recover programmer intent.  `getUserByEmail` might become
`["get", "User", "By", "Email"]` or `["g", "et", "User", "By", "Email"]`
depending on vocabulary decisions.

The regex approach is deterministic and reflects actual programmer conventions.
It recovers the programmer's intended noun tokens reliably.

### Usage in the pipeline

`split_identifier` is called by `TermEnricher` on the raw token before any
LLM call.  If the token is already a known English word (length ≥ 5, no
camelCase), the LLM expansion step is skipped and the token is used as-is —
saving an LLM call.

---

## 4. TermEnricher — offline knowledge extraction

```
src/code_indexer/graph/term_enricher.py
```

> **Source:** Inspired by the *offline knowledge-extraction stage* of
> **RepoLens** (see [§10](#10-sources-and-prior-work)).  RepoLens builds a
> term knowledge base from identifier names and their contexts before any
> query is processed.

### Design goal

Produce a **term knowledge base** — a persistent dictionary that maps
abbreviated code identifiers to their full English forms and definitions.
This is built once at index time and used at query time without LLM cost.

### The two-prompt protocol

For each unique token extracted by `split_identifier` from a symbol's name,
`TermEnricher` makes up to two LLM calls:

**Prompt 1 — Expansion**

```
You are a code documentation assistant.
Code context:
<function body, truncated to 300 tokens>

The identifier `{token}` appears in the code above.
Expand it to its full English noun phrase (2–5 words) based on how it is
used in context. Reply with ONLY the expanded phrase, no punctuation.
```

The response is expected to be 2–5 words.  The temperature is set to 0.0
for deterministic output (abbreviation expansion is not creative).

**Prompt 2 — Definition**

```
Define the software concept `{expanded_phrase}` in one sentence of at most
20 words. Describe what it IS in general software engineering terms — not
how it is used in any specific codebase.
```

Again temperature=0.0.  The definition must be general (codebase-agnostic)
because it will be used across many functions.

### The EnrichedTerm dataclass

```python
@dataclass
class EnrichedTerm:
    raw_name:      str        # "req"
    expanded_name: str        # "HTTP request"
    definition:    str        # "An incoming message from a client containing..."
    noun_tokens:   list[str]  # ["http", "request"]  ← split from expanded_name
```

`noun_tokens` is derived by running `split_identifier` on `expanded_name`.
These tokens are what actually get matched against a user's query keywords
in `TermKnowledgeBase.search_for_query()`.

### The TermKnowledgeBase

```python
@dataclass
class TermEntry:
    term:            EnrichedTerm
    symbol_summaries: dict[str, str]   # symbol_id → per-function summary

class TermKnowledgeBase:
    def add(self, term: EnrichedTerm, symbol_id: str, summary: str) -> None: ...
    def search_for_query(self, query: str) -> list[TermEntry]: ...
    def save(self, path: str) -> None: ...

    @classmethod
    def load(cls, path: str) -> "TermKnowledgeBase": ...
```

`search_for_query` splits the query into tokens with `split_identifier`,
then returns all `TermEntry` objects whose `noun_tokens` overlap with the
query tokens.  This is a fast set-intersection operation — no LLM needed.

### Caching and amortisation

Every `(raw_name, context_hash)` pair is cached before the LLM call.  If the
same abbreviation appears in many functions (e.g. `req` in a web framework),
the LLM is called once and the result is reused everywhere.

The knowledge base is serialised to `.term_kb.json` (or a configurable path).
Subsequent indexer runs load this file and only call the LLM for tokens that
are not already in the cache.

---

## 5. TermChunk — term-centric functionality summarisation

```
src/code_indexer/graph/term_chunker.py
```

> **Source:** Inspired by the *term-centric functionality summarisation*
> stage of **RepoLens**.  For each (function, term) pair, RepoLens generates
> a focused summary describing only the term-relevant slice of the function.

### Design goal

Add a second, more focused representation to the vector index for each
`(symbol, term)` pair.  Rather than embedding the entire function body,
embed a short description of what the function does with respect to one
specific term.

### The TermChunk

```python
@dataclass
class TermChunk:
    symbol:  SymbolNode     # the function
    term:    EnrichedTerm   # the term this chunk is about
    summary: str            # LLM-generated, one-sentence summary

    def to_embedding_text(self) -> str:
        return (
            f"Term: {self.term.expanded_name}\n"
            f"Definition: {self.term.definition}\n"
            f"In {self.symbol.name} ({self.symbol.file_path}:{self.symbol.start_line}): "
            f"{self.summary}"
        )
```

The embedding text front-loads the term name and definition.  This means the
vector for a TermChunk is anchored to the concept rather than to the function's
full body — which makes it easier for a query about that concept to retrieve it.

### The extraction prompt

`TermChunkExtractor` constructs a prompt for each `(symbol, term)` pair:

```
You are a code documentation assistant.
Function name: {symbol.name}
File: {symbol.file_path}:{symbol.start_line}

Code:
<function body, truncated to 400 tokens>

Describe in one sentence (max 25 words) what this function does with respect
to "{term.expanded_name}" ({term.definition}).
If this function has nothing to do with {term.expanded_name}, reply NOT_RELATED.
```

If the LLM replies `NOT_RELATED`, no `TermChunk` is created.  This is the
natural pruning step — a function may have a parameter named `req` (a
token that became the term `"HTTP request"`) but that does not mean the
function's purpose is related to HTTP request handling.

### Indexing

TermChunks are embedded using the same embedding model as whole-function
chunks and stored in the same vector collection under a different `chunk_type`
metadata field.  Both types coexist in the index.

At query time, both are retrieved.  The retriever's re-ranking step
(Reciprocal Rank Fusion) combines their scores.  A function can appear
multiple times — once as its whole-function chunk and once (or more) as
TermChunks — and RRF naturally aggregates these signals.

### Why TermChunks don't replace whole-function chunks

Whole-function chunks answer *"what does this function do overall?"*
TermChunks answer *"what does this function do with respect to X?"*
Both question types are valid.  A query about `handle_payment` in general
should retrieve the whole-function chunk.  A query about `"discount logic"`
should retrieve the TermChunk for the `discount` term within `handle_payment`.
Having both in the index means both query types succeed.

---

## 6. ConcernClusterer — online concern grouping

```
src/code_indexer/graph/concern_clusterer.py
```

> **Source:** Inspired by the *online retrieval-and-ranking stage* of
> **RepoLens**, which groups retrieved TermChunks into named "concerns" —
> high-level conceptual groupings that reveal cross-file structure.

### Design goal

After retrieval, the user has a ranked list of TermChunks scattered across
many files.  `ConcernClusterer` groups these into 2–5 named clusters
(called *concerns*), where each cluster represents a coherent conceptual
theme that the retrieved code relates to.

### The two-model design

The clusterer uses two LLM clients deliberately:

```python
class ConcernClusterer:
    def __init__(self, llm: BaseLLMClient, cluster_llm: BaseLLMClient): ...
```

| Call | Model | Why |
|---|---|---|
| `cluster()` — group n items into 2-5 clusters | `cluster_llm` (strong model) | Requires real semantic reasoning; wrong groupings mislead the agent |
| `rank_concerns()` — sort clusters by relevance | `llm` (cheap model) | Simple list reordering, easily done by a small model |

Typically: `cluster_llm = gpt-4o`, `llm = gpt-4o-mini`.

### The clustering prompt

```python
def _build_cluster_prompt(
    self,
    query: str,
    chunks: list[tuple[TermChunk, float]],
) -> str:
    items = "\n".join(
        f"{i}. [{chunk.symbol.name}] re: {chunk.term.expanded_name}: {chunk.summary}"
        for i, (chunk, _) in enumerate(chunks)
    )
    return f"""You are a software architect analysing code retrieval results.

Query: "{query}"

Retrieved functionalities:
{items}

Group these {len(chunks)} items into 2 to 5 high-level concerns.
A concern is a coherent feature or cross-cutting aspect of the codebase.
Items from different files can belong to the same concern.

Reply in JSON only, with this schema:
[
  {{
    "name": "Short concern name (3-6 words)",
    "description": "One sentence describing what this concern covers.",
    "indices": [0, 3, 7]
  }}
]"""
```

### The Concern dataclass

```python
@dataclass
class Concern:
    name:        str
    description: str
    chunks:      list[tuple[TermChunk, float]]   # (chunk, score) pairs

    def to_prompt_block(self) -> str: ...
    def to_mermaid(self) -> str: ...
```

`to_mermaid()` generates a `flowchart LR` Mermaid diagram showing only the
symbols in this concern and their call edges — useful for providing the AI
with a topology view scoped to one concern.

### Ranking after clustering

After clustering, the list of concerns is re-ranked by relevance to the
query using the cheap model:

```python
def _rank_concerns(self, query: str, concerns: list[Concern]) -> list[Concern]:
    prompt = f"""Query: "{query}"

Concerns:
{chr(10).join(f"{i}. {c.name}: {c.description}" for i, c in enumerate(concerns))}

Return the indices of these concerns in order of relevance to the query
(most relevant first). Reply with a JSON list of integers only.
"""
    ...
```

This separates the two LLM calls cleanly: the expensive model does the
semantic clustering, the cheap model does the mechanical reordering.

### Error handling

The cluster prompt returns JSON.  The clusterer wraps the LLM call in a
`try/except` with a fallback: if the LLM returns invalid JSON or omits
required fields, `ConcernClusterer` returns a single fallback concern named
`"Retrieved results"` containing all chunks in score order.  This ensures
the downstream prompt is always valid even if clustering fails.

---

## 7. Concern blocks in the prompt

```
src/code_indexer/graph/context_packer.py
    → render_concern_block(concerns: list[Concern]) -> str
    → PackedContext.to_prompt_text() -> str
```

### Why concerns go at the top

The LLM processes its context window sequentially (left to right in
transformer attention, though not strictly causally for non-autoregressive
tasks like encoding).  Empirically, information presented early in the
prompt is more reliably used for high-level orientation and planning, while
information later in the prompt is used for specific detail retrieval.

Placing the concern block **first** means the AI forms a high-level
understanding of the code's structure before it reads the detailed symbol
list.  This improves the quality of the AI's response on complex,
multi-faceted queries.

### The render function

```python
def render_concern_block(concerns: list[Concern]) -> str:
    lines = [
        "=== Inferred concerns (use as guidance, not ground truth) ===\n"
    ]
    for concern in concerns:
        lines.append(concern.to_prompt_block())
    lines.append(
        "\nNote: these concerns are inferred automatically and may be "
        "incomplete.\nUse them as starting points but continue to explore "
        "the codebase independently if needed.\n"
    )
    return "\n".join(lines)
```

The disclaimer at the end ("use as starting points") is critical.  Without
it, LLMs tend to treat the provided structure as authoritative and stop
searching outside it.  With it, the LLM treats the concerns as orientation
rather than constraint.

### PackedContext.to_prompt_text()

```python
# PackedContext returned by ContextPacker.pack()
text = packed.to_prompt_text(
    concerns=concerns,          # optional; if omitted, only the symbol list is rendered
    include_mermaid=True,       # whether to include a topology diagram
)
```

The output structure is:
```
[concern block]           ← from render_concern_block()
[mermaid diagram]         ← from to_mermaid()
[ranked symbol list]      ← from to_context_text()
```

All three sections are optional.  The concern block and mermaid diagram
together give the AI both semantic orientation (what the code is about) and
structural orientation (how the code calls each other).

---

## 8. LLM client layer — swappable backends

```
src/code_indexer/llm/
    base.py            → BaseLLMClient
    openai_client.py   → OpenAILLMClient
    ollama_client.py   → OllamaLLMClient
    __init__.py        → make_llm_client(), make_cluster_llm_client()
```

### The interface

```python
class BaseLLMClient(ABC):
    @abstractmethod
    def complete(self, prompt: str, *, temperature: float = 0.0) -> str:
        """Send a prompt, return the text response."""
```

That's the entire interface.  One method, two parameters.

### OpenAILLMClient

```python
client = OpenAILLMClient(
    api_key=settings.llm.api_key,
    model=settings.llm.model,          # e.g. "gpt-4o-mini"
)
```

- Uses `openai` Python library (`v1+` interface)
- `temperature=0.0` by default for all enrichment calls
- Retries on `RateLimitError` with exponential back-off: 2s, 4s, 8s, 16s
  (up to 4 retries)
- No retry on `AuthenticationError` or `InvalidRequestError` (not transient)

### OllamaLLMClient

```python
client = OllamaLLMClient(
    base_url=settings.llm.base_url,    # default: "http://localhost:11434"
    model=settings.llm.model,          # e.g. "llama3.2"
)
```

- Uses `httpx` to POST to Ollama's `/api/generate` endpoint
- Same retry logic as OpenAI client (Ollama can also be overloaded)
- No API key required

### Factory functions

```python
from code_indexer.llm import make_llm_client, make_cluster_llm_client
from code_indexer.core.config import get_settings

settings = get_settings()

# Cheap model for bulk operations
llm = make_llm_client(settings.llm)

# Strong model for concern clustering
cluster_llm = make_cluster_llm_client(settings.llm)
```

`make_cluster_llm_client` reads `settings.llm.cluster_model` (defaults to
the same as `model` if not set).  If the provider is `openai` and the cluster
model is different from the main model, it instantiates a second
`OpenAILLMClient` with the cluster model.

### Configuration

```
# .env or environment variables
LLM_PROVIDER=openai           # or "ollama"
LLM_MODEL=gpt-4o-mini         # main model (bulk operations)
LLM_CLUSTER_MODEL=gpt-4o      # clustering model (optional, defaults to LLM_MODEL)
LLM_API_KEY=sk-...            # required for openai
LLM_BASE_URL=http://...       # required for ollama
```

### Testing with a mock client

```python
class EchoLLMClient(BaseLLMClient):
    """Returns the prompt back — useful for testing prompt construction."""
    def complete(self, prompt: str, *, temperature: float = 0.0) -> str:
        return prompt

class FixedLLMClient(BaseLLMClient):
    """Always returns a fixed response — useful for testing downstream parsing."""
    def __init__(self, response: str):
        self.response = response
    def complete(self, prompt: str, *, temperature: float = 0.0) -> str:
        return self.response
```

Because `BaseLLMClient` is an abstract class, any test can inject a mock
without touching any other part of the system.

---

## 9. Full data flow — end to end

### OFFLINE (index time)

```
1. GraphExtractor
   ├── parse every source file with tree-sitter
   ├── emit SymbolNodes (functions, classes, methods)
   ├── emit CALLS / IMPORTS / INHERITS_FROM / INJECTS / CONTAINS edges
   └── record decorators on each symbol

2. For each SymbolNode:
   a. AST chunker → whole-function text chunk
   b. whole-function chunk → embed → store in vector DB

3. TermEnricher (runs after graph extraction):
   a. For each SymbolNode:
      i.  split_identifier(symbol.name) → raw tokens
      ii. For each raw token:
          - if token in KB cache → skip LLM call
          - else → prompt 1 (expand) + prompt 2 (define) → EnrichedTerm
          - cache result in TermKnowledgeBase

4. TermChunkExtractor (runs after TermEnricher):
   a. For each (SymbolNode, EnrichedTerm) pair:
      i.  LLM prompt → one-sentence summary re: that term
      ii. if NOT_RELATED → skip
      iii. else → TermChunk
      iv. TermChunk.to_embedding_text() → embed → store in vector DB
          (alongside whole-function chunks, with chunk_type="term")

5. Katz centrality:
   a. Build adjacency matrix from graph edges
      (CALLS + INJECTS + INHERITS_FROM)
   b. Power iteration until convergence
   c. Store centrality scores alongside each SymbolNode

6. TermKnowledgeBase.save(".term_kb.json")
```

### ONLINE (query time)

```
User query
   │
   ├─ TermKnowledgeBase.search_for_query(query)
   │     → overlapping noun tokens → list[TermEntry]   (no LLM, fast)
   │
   ├─ Embed query → ANN search in vector DB
   │     → top-50 (SymbolNode or TermChunk, score) pairs
   │
   ├─ Graph expansion via HybridRetriever
   │     → fetch callers/callees of top results
   │     → combine with vector results via RRF
   │
   ├─ Score blending
   │     final_score = 0.6 × vector_score + 0.4 × katz_centrality
   │
   ├─ ConcernClusterer.cluster(query, term_chunks_with_scores)
   │     → cluster_llm (gpt-4o): groups into 2-5 Concern objects
   │     → llm (gpt-4o-mini): ranks concerns by relevance
   │
   └─ ContextPacker.pack(candidates, token_budget, concerns)
         → greedy knapsack: selects symbols within token budget
         → PackedContext.to_prompt_text()
               → concern block (soft orientation)
               → mermaid diagram (topology)
               → ranked symbol list
               → sent to the AI
```

---

## 10. Sources and prior work

### RepoLens

The `TermEnricher`, `TermChunkExtractor`, and `ConcernClusterer` features are
directly inspired by the **RepoLens** system.  RepoLens introduced the
three-stage pipeline of:

1. **Offline knowledge extraction** — building a term knowledge base from
   identifier names and their contexts.
2. **Term-centric functionality summarisation** — summarising what each
   function does with respect to each term.
3. **Online concern grouping** — clustering retrieved results into named
   concerns at query time.

The implementation here adapts these ideas to the existing graph-based
indexer architecture.  The code files carry explicit attribution:

```python
# term_enricher.py:3
# Inspired by the offline knowledge-extraction stage of RepoLens

# term_chunker.py:3
# Inspired by the term-centric functionality summarisation stage of RepoLens

# concern_clusterer.py:3
# Inspired by the online retrieval-and-ranking stage of RepoLens
```

### arXiv:2601.08773 — "Reliable Graph-RAG for Codebases"

The paper *"Reliable Graph-RAG for Codebases"* (DKB paper, arXiv:2601.08773)
showed that deterministic AST-derived graphs with `extends`/`implements`/
`injects` edges produce significantly better retrieval than embedding-only
approaches for code understanding tasks.

This influenced:

- The decision to add **INJECTS edges** to the graph (Section 6 of the
  guide).
- The inclusion of INJECTS edges in the **Katz centrality** calculation
  (in addition to CALLS and INHERITS_FROM).
- The overall hybrid retrieval architecture (graph expansion + vector search).

From `docs/impact_analysis.md`:
> *Inspired by the DKB paper (arXiv:2601.08773) which showed that `extends` /
> `implements` / `injects` edges produce a richer dependency graph.*

### r/LocalLLaMA — code-chopper thread

A Reddit thread on r/LocalLLaMA introduced the `code-chopper` library and
two ideas that were directly incorporated:

1. **Katz centrality for ranking** — the `entity_rank` use-case in
   `code-chopper` used Katz centrality to identify the most-called functions
   in a codebase, ranking them as high-priority for code review or AI context.

2. **Knapsack optimisation for context packing** — a commenter suggested:
   > *"knapsack optimization so you only load the optimal context for the
   > agent"*

   This comment directly motivated the `ContextPacker` implementation.

From `docs/context_packing.md`:
> *Inspiration: a comment in the r/LocalLLaMA code-chopper thread (2025)*

### Vocabulary gap — information retrieval literature

The vocabulary gap problem is well-documented in classical IR research
(Furnas et al., 1987; Voorhees, 1994).  The specific application to code
identifiers and the use of LLM-based expansion as the fix is the RepoLens
contribution.

The `split_identifier` function's regex approach is a common technique in
source code analysis tools (it appears in similar form in various IDE plugins,
code search engines, and static analysis tools).  The innovation here is
feeding its output to an LLM expansion call rather than a static synonym
dictionary.
