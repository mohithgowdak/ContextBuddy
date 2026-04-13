# Changelog

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
