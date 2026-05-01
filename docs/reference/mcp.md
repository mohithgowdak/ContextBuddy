# MCP Server (`contextbuddy-mcp`)

ContextBuddy ships an optional MCP server so MCP-capable clients can:

- compress context into a budgeted prompt
- optionally search a local codebase/knowledge base to gather initial context, then compress it

## Install

```bash
pip install "contextbuddy[mcp]"
```

## Commands (one page)

See `docs/commands.md` for a one-page table of CLI commands + MCP tools.

## First-time setup (recommended)

ContextBuddy’s MCP server works best when you do two things once:

- **Set an allowed root list** (safety)
- **Build an index** (speed + better retrieval)

### 1) Safety: restrict which folders can be searched/indexed

Set `CONTEXTBUDDY_ALLOWED_ROOTS` to the repo(s) you want MCP to access.

- **Windows (PowerShell)**:

```powershell
$env:CONTEXTBUDDY_ALLOWED_ROOTS="D:\artiMIND\ContextBuddy"
```

- **macOS/Linux (bash/zsh)**:

```bash
export CONTEXTBUDDY_ALLOWED_ROOTS="$HOME/projects/ContextBuddy"
```

### 2) Start the MCP server (stdio)

Run:

```bash
contextbuddy-mcp
```

## Quickstart: configure in an MCP client (Cursor / Claude Desktop)

Most MCP clients have a config file where you register servers. The exact location varies by client, but the config shape usually looks like this:

```json
{
  "mcpServers": {
    "contextbuddy": {
      "command": "contextbuddy-mcp",
      "env": {
        "CONTEXTBUDDY_ALLOWED_ROOTS": "D:\\artiMIND\\ContextBuddy"
      }
    }
  }
}
```

Notes:
- **Windows paths**: use double backslashes (`\\`) inside JSON strings.
- **Multiple repos**: separate them with `;` (Windows) or `:` (macOS/Linux). Example: `"D:\\repo1;D:\\repo2"`.
- **No allowlist set**: MCP will still work, but it’s less safe (it can search/index any `root` you pass).

### 3) Build an index (optional, but strongly recommended)

You have three retrieval modes:

- **Scan (no setup)**: `search_and_compress`
- **Graph (fast + incremental)**: `graph_*` (imports + Python symbol spans)
- **Vector + Graph (best quality)**: `vector_*` + `vector_graph_search_and_compress`

Recommended first-time workflow (best quality):

1. Call `vector_build(root="...")`
2. Call `graph_build(root="...")`
3. Use `vector_graph_search_and_compress(...)` for daily queries

## Optional: richer Python call-chain extraction (tree-sitter)

ContextBuddy can optionally build a higher-accuracy Python “codegraph” (call edges) using tree-sitter.

Install:

```bash
pip install "contextbuddy[codegraph]"
```

This enables:
- better file-to-file expansion using call relationships (when a codegraph index exists)
- optional `key_flows` output in `project_overview_and_compress` (when enabled)

## Most common use case: IDE / repo knowledge

This MCP server is designed for the “assistant in your IDE” workflow:

1. Search your repo / docs for relevant snippets
2. Compress those snippets into a strict token budget
3. Send the compressed prompt to your LLM

In practice, this prevents:
- blown context windows
- “random” answers caused by noisy context
- high token bills from dumping whole files

## Tools exposed

### `compress`

Compress raw context into a budgeted prompt (no LLM call).

Inputs:
- `user_prompt` (str)
- `context` (str or list[str])
- `max_context_tokens` (int)
- `min_relevance` (float)
- `conservative_mode` (bool)
- `include_entities_section` (bool)

Returns:
- `prompt` (str)
- `report` (dict)

### `search_kb`

Search a local folder (codebase / notes) and return line previews.

Inputs:
- `query` (str)
- `root` (str, default `"."`)
- `max_matches`, `max_files`, `max_bytes_per_file` (limits)
- `context_lines` (preview lines)
- `group_adjacent` (merge nearby hits in the same file; default true)

Returns:
- `matches`: list of `{path, line, preview}`

### `search_and_compress`

One-shot workflow: **search → assemble context → compress**.

Inputs:
- `user_prompt` (str)
- `kb_query` (optional str; defaults to `user_prompt`)
- `root` (str, default `"."`)
- search limits (`max_matches`, `max_files`, `max_bytes_per_file`, `context_lines`)
- `group_adjacent` (merge nearby hits in the same file; default true)
- compression knobs (`max_context_tokens`, `min_relevance`, `conservative_mode`, `include_entities_section`)

Returns:
- `prompt` (str)
- `report` (dict)
- `kb_matches` (raw match list used to build context)

### `graph_build`

Build a persistent repo graph index (stdlib-only).

This stores:
- Python **symbols** (function/class spans)
- Python + JS/TS **import edges** (best-effort)

Inputs:
- `root` (str)
- `index_dir` (optional str; where indexes are stored)
- `max_files`, `max_bytes_per_file` (limits)

Returns:
- index build stats

### `graph_update`

Incrementally update the graph index when files change.

Inputs:
- `root`, `index_dir`
- `max_files`, `max_bytes_per_file`
- `prune_deleted` (bool)

Returns:
- scan/update stats

### `graph_search`

Search the graph index (symbol/file matches, plus hop expansion).

Inputs:
- `query` (str)
- `root`, `index_dir`
- `top_k`
- `hop_limit` (how many import hops to expand)
- `include_imports`, `include_importers`

Returns:
- `matches`: list of `{kind, path, score, name, start_line, end_line, preview}`

### `graph_search_and_compress`

One-shot workflow: **graph retrieval → compress**.

Inputs:
- `user_prompt` (str)
- `graph_query` (optional str; defaults to `user_prompt`)
- `root`, `index_dir`
- retrieval knobs (`top_k`, `hop_limit`, `include_imports`, `include_importers`)
- compression knobs (`max_context_tokens`, `min_relevance`, `conservative_mode`, `include_entities_section`)

Returns:
- `prompt`, `report`
- `graph_matches` + count

### `vector_build`

Build a persistent vector index over repo chunks.

Inputs:
- `root`, `index_dir`
- `embedder_id` (default `"localhash"`; can be `"ollama"`, `"sbert"`, `"openai"`, `"gemini"`)
- `embedder_config` (dict; embedder-specific config like `model`, `base_url`, `dims`)
- `max_files`, `max_bytes_per_file`, `batch_size`

Returns:
- index build stats

### `vector_update`

Incrementally update the vector index when files change.

Inputs:
- same as `vector_build` plus `prune_deleted`

Returns:
- update stats

### `vector_search`

Search the vector index and return ranked chunk matches.

Inputs:
- `query`
- `root`, `index_dir`
- `embedder_id`, `embedder_config`
- `top_k`, `min_score`

Returns:
- `matches`: list of `{path, score, id, preview, metadata}`

### `vector_search_and_compress`

One-shot workflow: **vector retrieval → compress**.

Inputs:
- `user_prompt` (str)
- `vector_query` (optional str; defaults to `user_prompt`)
- `root`, `index_dir`
- `embedder_id`, `embedder_config`
- retrieval knobs (`top_k`, `min_score`)
- compression knobs (`max_context_tokens`, `min_relevance`, `conservative_mode`, `include_entities_section`)

Returns:
- `prompt`, `report`
- `vector_matches` + count

### `vector_graph_search_and_compress` (best for IDEs)

Best-quality one-shot workflow:

**vector seeds → graph expansion → compress**

Inputs:
- `user_prompt` (str)
- `query` (optional str; defaults to `user_prompt`)
- `root`, `index_dir`
- `embedder_id`, `embedder_config`
- vector knobs (`vector_top_k`, `vector_min_score`)
- graph knobs (`graph_hop_limit`, `include_imports`, `include_importers`)
- compression knobs (`max_context_tokens`, `min_relevance`, `conservative_mode`, `include_entities_section`)

Returns:
- `prompt`, `report`
- `vector_matches` + count
- `graph_matches` + count

### `project_overview_and_compress`

Single-call repo overview tool for agent UX:

- Uses whichever indexes exist (vector/graph; builds graph automatically by default)
- Adds shallow repo “manifest” context (README / package.json / pyproject.toml / prisma schema) so overviews include run steps
- Compresses everything into your token budget
Inputs (common):
- `user_prompt` (str)
- `root` (str)
- `auto_build_graph` (bool, default true)
- `include_manifest_files` (bool, default true)
- `include_structured` (bool, default true)

Returns:
- `prompt`, `report`
- `mode` (one of: `vector_graph`, `vector`, `graph`, `fallback`)
- `entrypoints`, `core_modules`, `index_status` (when `include_structured=true`)

## Recommended defaults (repo search)

For codebases, these defaults work well:
- `root="."` (workspace root)
- `max_matches=30–80`
- `context_lines=2` (gives enough surrounding meaning)
- `max_context_tokens=1200–2500` depending on model window

If you see misses, increase `max_matches` or `max_context_tokens` first.

## Recommended defaults (indexed IDE usage)

- **First-time**: run `vector_build` + `graph_build` once per repo
- **Daily**: run `vector_update` + `graph_update` occasionally (or when you pull)
- **Query tool**: use `vector_graph_search_and_compress`

Suggested knobs:
- `vector_top_k=15–30`
- `graph_hop_limit=1` (raise to 2 if you need deeper dependencies)
- `max_context_tokens=1200–2500`

