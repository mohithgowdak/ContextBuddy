# User Guide

This guide shows how to use ContextBuddy end-to-end, from raw documents to a budgeted prompt (and optionally to a full RAG pipeline).

## 1) Compress raw text (no LLM call)

```python
from contextbuddy import ContextEngine, ContextEngineConfig

engine = ContextEngine(ContextEngineConfig(max_context_tokens=2000, dev_mode=True))
final_prompt, report = engine.build_prompt(
    user_prompt="What are the payment terms?",
    context=huge_text,
)
```

## 2) Load a PDF and compress

```python
from contextbuddy import ContextEngine, ContextEngineConfig, load

chunks = load("contract.pdf")
engine = ContextEngine(ContextEngineConfig(max_context_tokens=2000, dev_mode=True))
final_prompt, report = engine.build_prompt(user_prompt="What are the payment terms?", context=chunks)
```

## 3) Multi-doc RAG (search → compress → call LLM)

```python
from contextbuddy import MemoryStore, Retriever, load

store = MemoryStore().add(load("./docs/"))
retriever = Retriever(store, max_context_tokens=3000, dev_mode=True)
answer = retriever.query("What are the payment terms?", llm_call=my_llm)
```

## 4) Full pipeline (one-liner)

```python
from contextbuddy import Pipeline

pipeline = Pipeline.from_directory("./docs/", dev_mode=True)
answer = pipeline.query("Summarize the contract", llm_call=my_llm)
```

## 5) Production knobs (quick)

- Need fewer misses? use `conservative_mode=True` and/or raise `max_context_tokens`.
- Need more shrink? lower `max_context_tokens` and/or raise `min_relevance`.
- Need true semantics? use the Embedding Levels section in `README.md`.

## 6) IDE / repo knowledge (MCP)

If your main use case is “ask my codebase”, use the MCP server to:

**search → assemble context → compress**

See `docs/reference/mcp.md` for the tool list and parameters.

