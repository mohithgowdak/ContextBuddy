# FAQ

## Will this hurt answer quality?
It can **if you prune too aggressively**. Use the benchmark gate and tune:
- `max_context_tokens` (bigger = safer, smaller = cheaper)
- `min_relevance` (lower = keep more, safer; higher = prune more)
- `conservative_mode=True` when correctness matters more than maximum reduction

If you want proof on your data, run:
```bash
python -m contextbuddy bench --gate --json bench-report.json
```

## What does ContextBuddy guarantee?
- **Regex-matched entity survival** (IDs/dates/URLs/etc.) is a hard guarantee.
- **No empty output** for non-empty input.
- **No mid-sentence cuts** in summarization.
- **Deterministic core**: same input + config → same output.

## How do I tune it quickly?
Start here:
- **Safer defaults**: `conservative_mode=True`, `min_relevance=0.05`, `max_context_tokens=1500–4000`
- **More aggressive**: `min_relevance=0.20–0.35`, smaller `max_context_tokens`

Rule of thumb: if you see misses, lower `min_relevance` first.

## Does my data leave my machine?
Not by default.

Core runs locally with **zero dependencies**. Data only leaves your machine if you explicitly enable optional networked components (e.g. `OpenAIEmbedder`) or you pass an `llm_call` that calls an external API.

## How is this different from LangChain/LlamaIndex?
Those frameworks orchestrate pipelines and retrieval. ContextBuddy is the **compression layer**:
- it reduces tokens before the LLM call
- it preserves entities and produces an ROI report

You can use ContextBuddy standalone, or drop it into LangChain/LlamaIndex as a “compress before model” step.

## Does it work with streaming?
Yes, in two ways:

1) **LLM streaming**: `engine.run(stream=True)` yields LLM chunks (if your `llm_call` streams).
2) **Compression streaming (UX)**: `engine.build_prompt_stream(...)` yields progress events, and the CLI supports `--stream` for a live progress bar.

## What if I want accurate token counting?
Default token counting is heuristic (`len(text)/4`), which is fast and dependency-free.

For exact OpenAI-compatible token counts:
```bash
pip install "contextbuddy[tiktoken]"
```

## What’s the fastest way to verify it works?
Run the smoke tests:
- Windows: `scripts/smoke_test.ps1`
- macOS/Linux: `scripts/smoke_test.sh`

