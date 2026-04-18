# `ContextEngine` (core compression API)

`ContextEngine` is the heart of ContextBuddy. It takes a **user question** and **raw context** (string or list of chunks), selects the highest-signal chunks, preserves critical entities, enforces a strict token budget, and returns a final prompt plus a report.

## Imports

```python
from contextbuddy import ContextEngine, ContextEngineConfig
```

## Quick usage (no LLM call)

```python
engine = ContextEngine(ContextEngineConfig(max_context_tokens=2000, dev_mode=True))
final_prompt, report = engine.build_prompt(
    user_prompt="What are the payment terms?",
    context=big_text_or_chunks,
)
```

## Usage (with your LLM call)

```python
result = engine.run(
    user_prompt="What are the payment terms?",
    context=big_text_or_chunks,
    llm_call=lambda prompt: client.responses.create(model="gpt-4o-mini", input=prompt),
)
```

Async:

```python
result = await engine.arun(
    user_prompt="What are the payment terms?",
    context=big_text_or_chunks,
    llm_call=lambda prompt: async_client.responses.create(model="gpt-4o-mini", input=prompt),
)
```

## `ContextEngineConfig`

Key fields:

- **`max_context_tokens`**: hard cap for selected context (default 4000).
- **`min_relevance`**: prune threshold; lower keeps more, higher prunes more.
- **`conservative_mode`**: safety mode (keeps more; fewer misses).
- **`dev_mode`**: prints ROI telemetry.
- **`include_entities_section`**: adds a `KeyEntities:` section to the prompt.
- **`chunk_min_chars` / `chunk_merge_under_chars`**: chunk coherence defaults.

## What `build_prompt()` returns

`build_prompt()` returns `(final_prompt: str, report: ContextReport)`.

`ContextReport` includes:

- tokens before/after
- `% reduction`
- estimated cost savings (based on pricing preset)
- entity list (filtered to avoid common PDF noise)
- chunk indices selected

## How it works internally (high-level)

1. Chunk the input (document-aware).
2. Score chunk relevance (default `HybridScorer`).
3. Extract entities (regex) and apply keep-list + neighbor context.
4. Enforce budget (greedy selection + sentence-safe summarization fallback).
5. Compose final prompt + report.

See `docs/reference/architecture.md` for the full dataflow.

