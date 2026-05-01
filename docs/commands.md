# Commands (one page)

This page lists the most common “what do I run?” commands (CLI) and “what do I call?” tools (MCP) in one place.

| Mode | What you run/call | Purpose |
|---|---|---|
| **CLI** | `python -m contextbuddy compress --prompt "..." --file context.txt --show-prompt` | Compress a local text file into a budgeted prompt (demo/smoke). |
| **CLI** | `python -m contextbuddy compress --prompt "..." --context "..." --show-prompt` | Compress inline context into a budgeted prompt (demo/smoke). |
| **CLI** | `python -m contextbuddy bench --gate --json bench-report.json` | Run quality gate benchmarks (ship confidence). |
| **MCP** | `contextbuddy-mcp` | Start the MCP server (stdio). |
| **MCP tool** | `compress` | Compress raw context into a budgeted prompt (no retrieval). |
| **MCP tool** | `search_kb` | Scan-search a local repo/KB and return line previews. |
| **MCP tool** | `search_and_compress` | Scan-search → assemble context → compress. |
| **MCP tool** | `graph_build` | Build persistent repo graph (imports + Python symbol spans). |
| **MCP tool** | `graph_update` | Incrementally update the repo graph after changes. |
| **MCP tool** | `graph_search` | Query the repo graph (plus hop expansion). |
| **MCP tool** | `graph_search_and_compress` | Graph retrieval → compress. |
| **MCP tool** | `vector_build` | Build persistent vector index over repo chunks (embed + store). |
| **MCP tool** | `vector_update` | Incrementally update vector index after changes. |
| **MCP tool** | `vector_search` | Query the vector index (fast “semantic-ish” retrieval). |
| **MCP tool** | `vector_search_and_compress` | Vector retrieval → compress. |
| **MCP tool** | `vector_graph_search_and_compress` | **Best for IDEs**: vector seeds → graph expansion → compress. |

