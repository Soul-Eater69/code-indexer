# Chunking Strategies — Deep Dive

This document explains the internal mechanics of each chunking strategy and
helps you choose the right one for your use case.

---

## What is a Chunk?

A **chunk** is a contiguous piece of source code with:

- A **stable deterministic ID** (SHA-256 of `file_id:start_byte:end_byte`)
- **Byte and line offsets** into the original file
- A **semantic type** (`function`, `class`, `method`, `snippet`, …)
- An optional **name** (function/class name, extracted from the AST)
- Optional **context** lines before/after for surrounding code

Chunks are the atomic unit of the index.  Each chunk gets one embedding vector.

---

## Strategy 1: ASTChunker (default)

### How it works

```
ParsedFile.ast_nodes (list of ASTNode)
    │
    ▼
For each node:
  text = source_bytes[node.start_byte : node.end_byte]
  type = _NODE_TYPE_MAP[node.type]   # function/class/method/…
  name = find identifier child node
  ─────────────────────────────────
  if word_count(text) < min_chunk_size:  discard
  if word_count(text) > max_chunk_size:  fallback → TokenChunker
  else:                                  emit Chunk
```

### AST node type → ChunkType mapping

| Tree-sitter node type | Language | ChunkType |
|---|---|---|
| `function_definition` | Python | `FUNCTION` |
| `async_function_def` | Python | `FUNCTION` |
| `class_definition` | Python | `CLASS` |
| `decorated_definition` | Python | `FUNCTION` |
| `function_declaration` | JS/TS | `FUNCTION` |
| `arrow_function` | JS/TS | `FUNCTION` |
| `class_declaration` | JS/TS | `CLASS` |
| `method_definition` | JS/TS | `METHOD` |
| `interface_declaration` | TS | `INTERFACE` |
| `enum_declaration` | TS | `ENUM` |
| `function_item` | Rust | `FUNCTION` |
| `impl_item` | Rust | `CLASS` |
| `trait_item` | Rust | `INTERFACE` |
| `struct_item` | Rust | `CLASS` |
| `function_declaration` | Go | `FUNCTION` |
| `method_declaration` | Go | `METHOD` |
| `method_declaration` | Java | `METHOD` |
| `class_declaration` | Java | `CLASS` |

### Context annotation

After emitting each chunk, we attach surrounding lines:

```
context_before = lines[max(0, start_line-1-N) : start_line-1]
context_after  = lines[end_line : min(total, end_line+N)]
```

where `N = PARSER_CONTEXT_LINES` (default 3).

Context is stored separately from `content` so the embedding only sees the
main chunk text, but context is available for display in the UI.

---

## Strategy 2: TokenChunker

### Algorithm

```
token_budget = chunk_size                    (default 512)
overlap_budget = chunk_overlap               (default 64)

buffer = []
for line in source_lines:
    if tokens(buffer + line) > token_budget and buffer:
        emit chunk(buffer)
        overlap_text = last overlap_budget tokens of buffer
        buffer = []
    buffer.append(line)

if buffer: emit chunk(buffer)
```

### Token counting

We use `tiktoken` with the `cl100k_base` vocabulary (same as GPT-4 / GPT-4o).
This ensures that "512 tokens" here means the same 512 tokens that the LLM
will see in the prompt.

If `tiktoken` is not installed, we fall back to `len(text) // 4`
(approximate: code averages ~4 characters per token).

### Overlap detail

The `chunk_overlap` parameter controls how many tokens of the previous chunk
are prepended to the next one.  This preserves context at boundaries:

```
Chunk 1: [def authenticate(token: str) -> bool:
              ...
              payload = decode_jwt(token)   ← last 64 tokens
         ]

Chunk 2: [              payload = decode_jwt(token)   ← overlap
              if payload["exp"] < time.time():
              ...
         ]
```

Without overlap, a reader of chunk 2 wouldn't know what `payload` is.

---

## Strategy 3: SlidingWindowChunker

The simplest strategy.  Splits by character count with a sliding window.

```
content = "..."    # 1000 chars
chunk_size = 200
step = chunk_size - chunk_overlap = 180

windows: [0:200], [180:380], [360:560], [540:740], [720:920], [900:1000]
```

With `split_on_newline=True` (default), each window's end is snapped to the
nearest preceding newline so the split never happens mid-word.

### When to use it

- For non-code files (Markdown, YAML, JSON configs).
- As the ultimate fallback when Tree-sitter grammars are unavailable.
- When you explicitly want deterministic chunk boundaries independent of
  the tokenizer used at inference time.
