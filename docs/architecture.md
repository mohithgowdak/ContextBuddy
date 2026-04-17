# Architecture

ContextBuddy is a **zero-dependency context compression middleware**. It sits *before* your LLM call and makes sure you send fewer tokens **without dropping critical information**.

## Data flow (end-to-end)

```
load() → Store → Retriever → Compressor (ContextEngine) → (optional Router/Cache) → Your LLM
```

### The core loop
1. **Chunk** raw context into coherent units (document-aware).
2. **Score** each chunk for relevance to the user’s question.
3. **Force-keep entities** (regex-matched IDs/dates/etc.) + neighbor context.
4. **Budget**: pack chunks under `max_context_tokens` (sentence-safe summarization fallback).
5. **Compose** the final prompt + return a **report** (tokens before/after, entities kept, indices selected).

## Module map (what lives where)

- **`contextbuddy/engine.py`**
  - `ContextEngine`: the compression API
  - `ContextReport`: ROI metrics returned with each run
  - `build_prompt_stream`: streaming events for UX (progress + live token stats)

- **`contextbuddy/chunking.py`**
  - `SmartChunker`: document-aware chunking (auto/legal/pdf/python-code)
  - `Chunker`: generic baseline chunking + merge rules

- **`contextbuddy/entities.py`**
  - `EntityExtractor`: regex-based entity detection (hard safety net)

- **`contextbuddy/budget.py`**
  - `BudgetEnforcer`: token budgeting + sentence-safe extractive summarization

- **`contextbuddy/store/`**
  - `MemoryStore`: zero-dep in-memory search
  - `PersistentStore`: JSON-backed persistence

- **`contextbuddy/retriever.py`**
  - `Retriever`: search → compress → call LLM (sync/async)

- **`contextbuddy/pipeline.py`**
  - `Pipeline`: loaders + store + retriever + optional router/caches

- **`contextbuddy/loaders/`**
  - `load()`: file/dir/url dispatcher
  - Optional deps behind guards (`pdf`, `web`, `docx`)

- **`contextbuddy/embedder.py` / `contextbuddy/types.py`**
  - `Embedder` / `AsyncEmbedder` protocols (duck-typed)
  - `LocalHashEmbedder` (zero-dep, deterministic)
  - Optional `OpenAIEmbedder` behind install guard

- **`contextbuddy/benchmarks.py`**
  - Benchmark harness + quality gate (`contextbuddy bench --gate`)

## Guarantees (red lines)

These are treated as **bugs** if violated:
- **Entity survival**: regex-matched entities must survive compression.
- **Never empty**: non-empty input must not produce empty output.
- **Sentence-safe**: no mid-sentence cuts in the final output.
- **Deterministic core**: same input + same config = same output.

## Optional dependencies (how we keep core zero-dep)

Core installs with:
```
pip install contextbuddy
```

Extras are opt-in and guarded with explicit errors:
- PDF: `pip install "contextbuddy[pdf]"`
- Web: `pip install "contextbuddy[web]"`
- DOCX: `pip install "contextbuddy[docx]"`
- OpenAI: `pip install "contextbuddy[openai]"`

## Failure modes + mitigations (what can go wrong)

- **Recall misses in relevance scoring**
  - Mitigation: conservative mode + entity keep-list + benchmark gate.

- **Long documents / latency spikes**
  - Mitigation: chunk coherence rules, budgeting, and (future) caching; measure p95 via `bench`.

- **PDF extraction artifacts**
  - Mitigation: PDF normalization + SmartChunker (avoid page-wise chunks).

## Quality gate (how we prevent regressions)

Run:
```bash
python -m contextbuddy bench --gate --json bench-report.json
```

This produces:
- answer survival rate (proxy)
- entity survival rate (hard requirement)
- mean reduction %
- latency mean + p95

See `docs/benchmarks.md`.

