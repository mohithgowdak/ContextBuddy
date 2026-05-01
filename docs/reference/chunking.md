# Chunking (why it matters)

Chunking quality is one of the three pillars of stripping efficiency:

- relevance scoring accuracy
- entity preservation
- **chunk boundary quality**

ContextBuddy provides both basic and document-aware chunkers.

## `SmartChunker` (recommended)

Defined in `src/contextbuddy/chunking.py`.

It supports `doc_type` modes:

- `pdf`: normalizes PDF line-break artifacts and then chunks coherently
- `legal` / `contract`: clause-aware grouping
- `code`: python-aware boundaries
- `auto`: detects legal/code patterns and chooses strategy

ContextEngine uses `SmartChunker` by default.

## Configuration knobs (via `ContextEngineConfig`)

- `chunk_min_chars` (default 100): coherence minimum
- `chunk_merge_under_chars` (default 200): merges small fragments

## PDF note

PDF loader extracts page text and then *re-chunks the full text* (not page-wise), using `SmartChunker(doc_type="pdf")`.

