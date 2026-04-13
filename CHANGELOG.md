# Changelog

## v0.1.0 (2026-04-13)

Initial release.

- **ContextEngine**: core pipeline with chunking, semantic scoring, pruning, entity keep-list, and token budgeting.
- **wrap_openai()**: drop-in wrapper that transparently compresses system messages in chat completions.
- **Async support**: `engine.arun()` for non-blocking LLM calls.
- **CLI**: `python -m contextbuddy compress` for instant demos without writing code.
- **Rich telemetry**: colored box-drawing ROI output in dev mode (the screenshot everyone will share).
- **Entity extraction**: emails, URLs, dates, UUIDs, tickets, phone numbers, money, IPs, version strings, ID-like values.
- **Pre-built pricing**: cost presets for GPT-4o, GPT-4.1, Claude, Gemini, and local models.
- **Zero dependencies**: core library works out of the box. Optional extras for tiktoken and OpenAI embeddings.
