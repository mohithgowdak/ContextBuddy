# ContextBuddy

Context compression middleware for LLM pipelines.

## Before writing any code
1. Read .cursor/rules/quality.mdc — the red lines
2. Read .cursor/rules/architecture.mdc — the module map
3. Ask: does this improve stripping efficiency?

## Current sprint (v0.3.0 — April 20)
- [ ] Pluggable embedders
- [ ] Async support  
- [ ] LangGraph node
- [ ] Streaming ROI
- [ ] MCP server
- [ ] PyPI publish

## Run before every commit
pytest tests/ -v
