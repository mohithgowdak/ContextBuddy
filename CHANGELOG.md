# Changelog

## v0.3.0 (2026-04-22)

### New: MCP Server (`contextbuddy.mcp`)

ContextBuddy now ships as an MCP server, making it directly usable inside Cursor,
Claude Desktop, and any MCP-capable client — no code changes required.

```bash
pip install "contextbuddy[mcp]"
contextbuddy-mcp   # starts the stdio server
```

**12 tools exposed:**
- `compress` — compress any raw context into a token-budgeted prompt
- `search_kb` / `search_and_compress` — keyword search + compress (no index needed)
- `graph_build` / `graph_update` / `graph_search` / `graph_search_and_compress` — Python symbol + import graph index
- `vector_build` / `vector_update` / `vector_search` / `vector_search_and_compress` — semantic vector index
- `vector_graph_search_and_compress` — best-quality: vector seeds → graph expansion → compress

**3 slash-command prompts:** `/cb`, `/cb_deep`, `/cb_index`

**Setup:** Copy `.cursor/mcp.json.example` to `.cursor/mcp.json`, set `CONTEXTBUDDY_ALLOWED_ROOTS`, restart Cursor.

---

### New: Pluggable Embedders

All embedders conform to a common `Embedder` protocol and are injectable at engine creation.
New `scorer=` parameter on `ContextEngine` to inject any custom scorer.

---

### New: HybridScorer (`contextbuddy.hybrid_scorer`)

- **BM25 scoring** — term-frequency saturation, doc-length normalization, IDF weighting
- **Lightweight stemmer** — "payments" matches "payment", "running" matches "run"
- **Built-in synonym thesaurus** — ~200 word groups covering business, legal, tech vocabulary
- **Character n-gram fuzzy matching** — catches typos and spelling variants
- Intent-aware bonuses: numeric density for metrics queries, method heading detection for "how does X work"
- Weighted combination: BM25 (70%) + synonym (15%) + n-gram (15%), all configurable

---

### New: Streaming ROI (`build_prompt_stream`)

`engine.build_prompt_stream()` yields `CompressionEvent` objects:
`start` → `chunked` → `scoring` → `scored` (with report) → `done` (with prompt + report)

---

### Bug Fixes

- **engine.py** — fixed sync/streaming entity mismatch; extracted `_compress()` eliminating
  150 lines of duplication; fixed conservative mode config mutation; fixed streaming stage
  ordering (`scoring` fires before compression, `scored` after with real report)
- **budget.py** — fixed O(n²) token counting → O(n) with incremental `running_total`
- **chunking.py** — fixed trailing fragments silently dropped; now merged into previous chunk
- **hybrid_scorer.py** — n-gram exact matches skip inner loop via set lookup
- **cache.py** — `CachedEmbedder` now validates embedder response length; removed `type: ignore`
- **index/vector.py** — `flush()` in `build()` and `update()` validates vector count to prevent
  silent chunk loss from short embedder responses
- **router.py** — `"summarize"` moved to complex keywords; added imperative task verb detection
  ("summarize the PDF", "debug this" now score as complex)
- **benchmarks.py** — fixed unclosed file handle in `load_dataset()`

---

### Changed

- `[all]` extra now includes `mcp[cli]`
- `.cursor/mcp.json` is gitignored; `.cursor/mcp.json.example` committed as setup template

---

### Async note

`engine.arun()` is async-compatible for the LLM call. Compression runs synchronously.
For high-concurrency workloads: `asyncio.to_thread(engine.build_prompt, ...)`.
True async compression planned for v0.4.0.

---

### Backward Compatibility

All existing APIs unchanged. `SemanticScorer` and `LocalHashEmbedder` remain fully supported.

---


## v0.2.0 (2026-04-13)

Full-stack context pipeline. ContextBuddy is now a complete LangChain alternative.

### New: Document Loaders (`contextbuddy.loaders`)
- Unified `load()` dispatcher: auto-detects PDFs, URLs, DOCX, text files, CSV, JSON, directories.
- Batch loading: `load(["a.pdf", "b.txt", "https://..."])`.
- Directory loader: recursive, with source-file prefixing and configurable depth/size limits.

### New: Vector Store (`contextbuddy.store`)
- `MemoryStore`: in-memory vector index with semantic search, auto-deduplication, and metadata.
- `PersistentStore`: JSON-file-backed store that survives restarts.
- Pure-Python cosine search (zero deps, fast enough for <100k chunks).

### New: Retriever (`contextbuddy.retriever`)
- `Retriever`: one-object multi-doc RAG. Search store -> compress -> call LLM.
- Async support via `aquery()`.

### New: Pipeline (`contextbuddy.pipeline`)
- `Pipeline`: wires loaders + store + retriever + compressor + router + cache.
- `Pipeline.from_directory()`: one-liner setup.
- `Pipeline.from_sources()`: load multiple files/URLs/dirs at once.

### New: Smart Router (`contextbuddy.router`)
- `Router`: route queries to cheap or expensive models based on complexity.
- `score_complexity()`: offline heuristic scorer (keyword + length + structure analysis).

### New: Caching (`contextbuddy.cache`)
- `EmbeddingCache`: avoid re-embedding the same text (in-memory + file persistence).
- `ResponseCache`: avoid redundant LLM calls (TTL-based expiry).
- `CachedEmbedder`: drop-in wrapper for any embedder.

### New: Streaming
- `engine.run(stream=True)`: emit ROI report, then yield LLM chunks.
- `engine.arun(stream=True)`: async streaming.

### New: Agent Tools (`contextbuddy.tools`)
- `make_search_tool(store)`: OpenAI function-calling compatible document search.
- `make_compress_tool(engine)`: OpenAI function-calling compatible context compression.
- `handle_tool_call()`: dispatcher for tool call responses.

---

## v0.1.0 (2026-04-13)

Initial release.

- **ContextEngine**: core pipeline with chunking, semantic scoring, pruning, entity keep-list, and token budgeting.
- **wrap_openai()**: drop-in wrapper that transparently compresses system messages in chat completions.
- **Async support**: `engine.arun()` for non-blocking LLM calls.
- **CLI**: `python -m contextbuddy compress` for instant demos without writing code.
- **Rich telemetry**: colored box-drawing ROI output in dev mode.
- **Entity extraction**: emails, URLs, dates, UUIDs, tickets, phone numbers, money, IPs, version strings, ID-like values.
- **Pre-built pricing**: cost presets for GPT-4o, GPT-4.1, Claude, Gemini, and local models.
- **Zero dependencies**: core library works out of the box.
