from __future__ import annotations

"""
ContextBuddy MCP server.

This exposes ContextBuddy as an MCP tool provider for token reduction:

- optionally search your codebase/knowledge base to gather initial context
- compress that context into a budgeted prompt for your LLM call

Install:
    pip install "contextbuddy[mcp]"

Run (stdio):
    contextbuddy-mcp
"""

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Optional


def _require_mcp():
    try:
        from mcp.server.fastmcp import FastMCP  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "ContextBuddy MCP server requires the 'mcp' package. "
            "Install with: pip install \"contextbuddy[mcp]\""
        ) from e
    return FastMCP


def create_server():
    FastMCP = _require_mcp()

    from ..engine import ContextEngine, ContextEngineConfig
    from .kb import matches_to_context, search_codebase
    from .security import validate_root
    from ..index.graph import RepoGraphIndex, build_default_index_dir
    from ..index.vector import RepoVectorIndex

    mcp = FastMCP("ContextBuddy", json_response=True)

    @mcp.tool()
    def compress(
        user_prompt: str,
        context: str | Sequence[str],
        max_context_tokens: int = 2000,
        min_relevance: float = 0.15,
        conservative_mode: bool = False,
        include_entities_section: bool = True,
    ) -> Dict[str, Any]:
        """Compress context to fit a token budget. Use this BEFORE sending context to any LLM.

        WHEN TO USE: Whenever you have gathered context (file contents, search results,
        documentation) and need to reduce it before an LLM call. This prevents blown
        context windows, saves tokens/cost, and improves answer quality by removing noise.

        Returns a compressed prompt string and a report with token counts and savings."""
        engine = ContextEngine(
            ContextEngineConfig(
                max_context_tokens=int(max_context_tokens),
                min_relevance=float(min_relevance),
                conservative_mode=bool(conservative_mode),
                dev_mode=False,
                include_entities_section=bool(include_entities_section),
            )
        )
        prompt, report = engine.build_prompt(user_prompt=str(user_prompt), context=context)
        return {"prompt": prompt, "report": asdict(report)}

    @mcp.tool()
    def search_kb(
        query: str,
        root: str = ".",
        max_matches: int = 30,
        max_files: int = 500,
        max_bytes_per_file: int = 512_000,
        context_lines: int = 1,
        case_sensitive: bool = False,
        group_adjacent: bool = True,
    ) -> Dict[str, Any]:
        """Search files in a local codebase or knowledge base for a query.

        WHEN TO USE: When the user asks about code, documentation, or project files
        and you need to find relevant snippets. Returns line-based previews with file
        paths. Use this as the first step before compressing results.

        Searches file contents using keyword matching with context lines around hits."""
        rootp = validate_root(root)
        matches = search_codebase(
            query=str(query),
            root=str(rootp),
            max_matches=int(max_matches),
            max_files=int(max_files),
            max_bytes_per_file=int(max_bytes_per_file),
            context_lines=int(context_lines),
            case_sensitive=bool(case_sensitive),
            group_adjacent=bool(group_adjacent),
        )
        return {"matches": [asdict(m) for m in matches], "count": len(matches)}

    @mcp.tool()
    def search_and_compress(
        user_prompt: str,
        kb_query: Optional[str] = None,
        root: str = ".",
        max_matches: int = 30,
        max_files: int = 500,
        max_bytes_per_file: int = 512_000,
        context_lines: int = 1,
        group_adjacent: bool = True,
        max_context_tokens: int = 2000,
        min_relevance: float = 0.15,
        conservative_mode: bool = False,
        include_entities_section: bool = True,
    ) -> Dict[str, Any]:
        """Search a codebase and compress the results into a token-budgeted prompt in one step.

        WHEN TO USE: This is the **recommended default tool** for answering questions
        about a codebase or repo. It searches for relevant code/docs, then compresses
        the results to fit a token budget. Use this whenever the user asks about code,
        architecture, bugs, or documentation in their project.

        Combines search_kb + compress into a single call. No LLM call is made."""
        rootp = validate_root(root)
        q = str(kb_query) if kb_query else str(user_prompt)
        matches = search_codebase(
            query=q,
            root=str(rootp),
            max_matches=int(max_matches),
            max_files=int(max_files),
            max_bytes_per_file=int(max_bytes_per_file),
            context_lines=int(context_lines),
            case_sensitive=False,
            group_adjacent=bool(group_adjacent),
        )
        context_chunks = matches_to_context(matches)

        engine = ContextEngine(
            ContextEngineConfig(
                max_context_tokens=int(max_context_tokens),
                min_relevance=float(min_relevance),
                conservative_mode=bool(conservative_mode),
                dev_mode=False,
                include_entities_section=bool(include_entities_section),
            )
        )
        prompt, report = engine.build_prompt(user_prompt=str(user_prompt), context=context_chunks)
        return {
            "prompt": prompt,
            "report": asdict(report),
            "kb_matches": [asdict(m) for m in matches],
            "kb_match_count": len(matches),
        }

    @mcp.tool()
    def graph_build(
        root: str = ".",
        index_dir: Optional[str] = None,
        max_files: int = 50_000,
        max_bytes_per_file: int = 512_000,
    ) -> Dict[str, Any]:
        """Build a repo graph index for fast code navigation and dependency tracking.

        WHEN TO USE: Run this once per repo before using graph_search or
        graph_search_and_compress. It indexes Python symbols (functions, classes)
        and import edges. Re-run after major code changes or use graph_update for
        incremental updates."""
        rootp = validate_root(root)
        idx_dir = Path(index_dir).resolve() if index_dir else build_default_index_dir()
        idx = RepoGraphIndex(root=rootp, index_dir=idx_dir)
        stats = idx.build(max_files=int(max_files), max_bytes_per_file=int(max_bytes_per_file))
        return {"root": str(rootp), **stats}

    @mcp.tool()
    def graph_update(
        root: str = ".",
        index_dir: Optional[str] = None,
        max_files: int = 50_000,
        max_bytes_per_file: int = 512_000,
        prune_deleted: bool = True,
    ) -> Dict[str, Any]:
        """Incrementally update the repo graph index after file changes.

        WHEN TO USE: After editing files or pulling new code, run this instead of
        a full graph_build. Only re-indexes changed files."""
        rootp = validate_root(root)
        idx_dir = Path(index_dir).resolve() if index_dir else build_default_index_dir()
        idx = RepoGraphIndex(root=rootp, index_dir=idx_dir)
        stats = idx.update(
            max_files=int(max_files),
            max_bytes_per_file=int(max_bytes_per_file),
            prune_deleted=bool(prune_deleted),
        )
        return {"root": str(rootp), **stats}

    @mcp.tool()
    def graph_search(
        query: str,
        root: str = ".",
        index_dir: Optional[str] = None,
        top_k: int = 20,
        hop_limit: int = 1,
        include_imports: bool = True,
        include_importers: bool = False,
        max_preview_lines: int = 80,
    ) -> Dict[str, Any]:
        """Search the repo graph index for symbols, functions, classes, and their dependencies.

        WHEN TO USE: When you need to find specific code symbols, understand import
        relationships, or trace dependencies. Supports hop expansion to follow imports.
        Requires graph_build to have been run first."""
        rootp = validate_root(root)
        idx_dir = Path(index_dir).resolve() if index_dir else build_default_index_dir()
        idx = RepoGraphIndex(root=rootp, index_dir=idx_dir)
        matches = idx.search(
            query=str(query),
            top_k=int(top_k),
            hop_limit=int(hop_limit),
            include_imports=bool(include_imports),
            include_importers=bool(include_importers),
            max_preview_lines=int(max_preview_lines),
        )
        return {"matches": [asdict(m) for m in matches], "count": len(matches)}

    @mcp.tool()
    def graph_search_and_compress(
        user_prompt: str,
        graph_query: Optional[str] = None,
        root: str = ".",
        index_dir: Optional[str] = None,
        top_k: int = 25,
        hop_limit: int = 1,
        include_imports: bool = True,
        include_importers: bool = False,
        max_preview_lines: int = 120,
        max_context_tokens: int = 2000,
        min_relevance: float = 0.15,
        conservative_mode: bool = False,
        include_entities_section: bool = True,
    ) -> Dict[str, Any]:
        """Search for code using the graph index and compress results into a token-budgeted prompt.

        WHEN TO USE: When you need to answer questions about code architecture,
        function implementations, or dependency chains. Better than search_and_compress
        for navigating code structure. Requires graph_build to have been run first."""
        rootp = validate_root(root)
        idx_dir = Path(index_dir).resolve() if index_dir else build_default_index_dir()
        idx = RepoGraphIndex(root=rootp, index_dir=idx_dir)
        q = str(graph_query) if graph_query else str(user_prompt)
        matches = idx.search(
            query=q,
            top_k=int(top_k),
            hop_limit=int(hop_limit),
            include_imports=bool(include_imports),
            include_importers=bool(include_importers),
            max_preview_lines=int(max_preview_lines),
        )
        context_chunks = idx.matches_to_context(matches)

        engine = ContextEngine(
            ContextEngineConfig(
                max_context_tokens=int(max_context_tokens),
                min_relevance=float(min_relevance),
                conservative_mode=bool(conservative_mode),
                dev_mode=False,
                include_entities_section=bool(include_entities_section),
            )
        )
        prompt, report = engine.build_prompt(user_prompt=str(user_prompt), context=context_chunks)
        return {
            "prompt": prompt,
            "report": asdict(report),
            "graph_matches": [asdict(m) for m in matches],
            "graph_match_count": len(matches),
        }

    @mcp.tool()
    def vector_build(
        root: str = ".",
        index_dir: Optional[str] = None,
        embedder_id: str = "localhash",
        embedder_config: Optional[Dict[str, Any]] = None,
        max_files: int = 50_000,
        max_bytes_per_file: int = 512_000,
        batch_size: int = 64,
    ) -> Dict[str, Any]:
        """Build a semantic vector index over repo files for similarity-based code search.

        WHEN TO USE: Run this once per repo for best search quality. Enables semantic
        matching (finds relevant code even when keywords differ). Supports multiple
        embedders: localhash (zero-dep default), ollama, sbert, openai, gemini."""
        rootp = validate_root(root)
        idx_dir = Path(index_dir).resolve() if index_dir else build_default_index_dir()
        idx = RepoVectorIndex(
            root=rootp,
            index_dir=idx_dir,
            embedder_id=str(embedder_id),
            embedder_config=dict(embedder_config or {}),
        )
        stats = idx.build(
            max_files=int(max_files),
            max_bytes_per_file=int(max_bytes_per_file),
            batch_size=int(batch_size),
        )
        return {"root": str(rootp), **stats}

    @mcp.tool()
    def vector_update(
        root: str = ".",
        index_dir: Optional[str] = None,
        embedder_id: str = "localhash",
        embedder_config: Optional[Dict[str, Any]] = None,
        max_files: int = 50_000,
        max_bytes_per_file: int = 512_000,
        batch_size: int = 64,
        prune_deleted: bool = True,
    ) -> Dict[str, Any]:
        """Incrementally update the vector index after file changes.

        WHEN TO USE: After editing files or pulling new code, run this instead of
        a full vector_build. Only re-embeds changed files."""
        rootp = validate_root(root)
        idx_dir = Path(index_dir).resolve() if index_dir else build_default_index_dir()
        idx = RepoVectorIndex(
            root=rootp,
            index_dir=idx_dir,
            embedder_id=str(embedder_id),
            embedder_config=dict(embedder_config or {}),
        )
        stats = idx.update(
            max_files=int(max_files),
            max_bytes_per_file=int(max_bytes_per_file),
            batch_size=int(batch_size),
            prune_deleted=bool(prune_deleted),
        )
        return {"root": str(rootp), **stats}

    @mcp.tool()
    def vector_search(
        query: str,
        root: str = ".",
        index_dir: Optional[str] = None,
        embedder_id: str = "localhash",
        embedder_config: Optional[Dict[str, Any]] = None,
        top_k: int = 20,
        min_score: float = 0.0,
        max_preview_chars: int = 900,
    ) -> Dict[str, Any]:
        """Search the vector index for semantically similar code chunks.

        WHEN TO USE: When keyword search misses relevant results because the code
        uses different terminology. Finds code by meaning, not just exact words.
        Requires vector_build to have been run first."""
        rootp = validate_root(root)
        idx_dir = Path(index_dir).resolve() if index_dir else build_default_index_dir()
        idx = RepoVectorIndex(
            root=rootp,
            index_dir=idx_dir,
            embedder_id=str(embedder_id),
            embedder_config=dict(embedder_config or {}),
        )
        matches = idx.search(
            query=str(query),
            top_k=int(top_k),
            min_score=float(min_score),
            max_preview_chars=int(max_preview_chars),
        )
        return {"matches": [asdict(m) for m in matches], "count": len(matches)}

    @mcp.tool()
    def vector_search_and_compress(
        user_prompt: str,
        vector_query: Optional[str] = None,
        root: str = ".",
        index_dir: Optional[str] = None,
        embedder_id: str = "localhash",
        embedder_config: Optional[Dict[str, Any]] = None,
        top_k: int = 25,
        min_score: float = 0.0,
        max_preview_chars: int = 1200,
        max_context_tokens: int = 2000,
        min_relevance: float = 0.15,
        conservative_mode: bool = False,
        include_entities_section: bool = True,
    ) -> Dict[str, Any]:
        """Search for code using the vector index and compress results into a token-budgeted prompt.

        WHEN TO USE: When you need semantic search + compression in one step. Better
        than keyword search for finding code that uses different terminology than the
        query. Requires vector_build to have been run first."""
        rootp = validate_root(root)
        idx_dir = Path(index_dir).resolve() if index_dir else build_default_index_dir()
        idx = RepoVectorIndex(
            root=rootp,
            index_dir=idx_dir,
            embedder_id=str(embedder_id),
            embedder_config=dict(embedder_config or {}),
        )
        q = str(vector_query) if vector_query else str(user_prompt)
        matches = idx.search(
            query=q,
            top_k=int(top_k),
            min_score=float(min_score),
            max_preview_chars=int(max_preview_chars),
        )
        context_chunks = idx.matches_to_context(matches)

        engine = ContextEngine(
            ContextEngineConfig(
                max_context_tokens=int(max_context_tokens),
                min_relevance=float(min_relevance),
                conservative_mode=bool(conservative_mode),
                dev_mode=False,
                include_entities_section=bool(include_entities_section),
            )
        )
        prompt, report = engine.build_prompt(user_prompt=str(user_prompt), context=context_chunks)
        return {
            "prompt": prompt,
            "report": asdict(report),
            "vector_matches": [asdict(m) for m in matches],
            "vector_match_count": len(matches),
        }

    @mcp.tool()
    def vector_graph_search_and_compress(
        user_prompt: str,
        query: Optional[str] = None,
        root: str = ".",
        index_dir: Optional[str] = None,
        embedder_id: str = "localhash",
        embedder_config: Optional[Dict[str, Any]] = None,
        vector_top_k: int = 20,
        vector_min_score: float = 0.0,
        graph_hop_limit: int = 1,
        include_imports: bool = True,
        include_importers: bool = False,
        max_preview_chars: int = 900,
        max_preview_lines: int = 80,
        max_context_tokens: int = 2000,
        min_relevance: float = 0.15,
        conservative_mode: bool = False,
        include_entities_section: bool = True,
    ) -> Dict[str, Any]:
        """BEST QUALITY: Hybrid vector + graph search, then compress into a token-budgeted prompt.

        WHEN TO USE: This is the **highest quality retrieval tool**. Use this for
        complex questions about code architecture, debugging, or understanding how
        modules connect. It combines semantic search (vector) with dependency tracking
        (graph) for the most complete context. Requires both vector_build and
        graph_build to have been run first.

        Workflow: vector seeds -> graph hop expansion -> dedupe -> compress."""
        rootp = validate_root(root)
        idx_dir = Path(index_dir).resolve() if index_dir else build_default_index_dir()

        q = str(query) if query else str(user_prompt)

        v = RepoVectorIndex(
            root=rootp,
            index_dir=idx_dir,
            embedder_id=str(embedder_id),
            embedder_config=dict(embedder_config or {}),
        )
        v_matches = v.search(
            query=q,
            top_k=int(vector_top_k),
            min_score=float(vector_min_score),
            max_preview_chars=int(max_preview_chars),
        )
        v_context = v.matches_to_context(v_matches)

        g = RepoGraphIndex(root=rootp, index_dir=idx_dir)
        g_matches = g.expand_from_files(
            [m.path for m in v_matches],
            hop_limit=int(graph_hop_limit),
            include_imports=bool(include_imports),
            include_importers=bool(include_importers),
            top_k=max(10, int(vector_top_k) * 2),
            max_preview_lines=int(max_preview_lines),
        )
        g_context = g.matches_to_context(g_matches)

        # combine + dedupe while preserving order (vector seeds first)
        seen = set()
        context_chunks: List[str] = []
        for ch in [*v_context, *g_context]:
            key = ch.strip()
            if not key or key in seen:
                continue
            seen.add(key)
            context_chunks.append(ch)

        engine = ContextEngine(
            ContextEngineConfig(
                max_context_tokens=int(max_context_tokens),
                min_relevance=float(min_relevance),
                conservative_mode=bool(conservative_mode),
                dev_mode=False,
                include_entities_section=bool(include_entities_section),
            )
        )
        prompt, report = engine.build_prompt(user_prompt=str(user_prompt), context=context_chunks)
        return {
            "prompt": prompt,
            "report": asdict(report),
            "vector_matches": [asdict(m) for m in v_matches],
            "vector_match_count": len(v_matches),
            "graph_matches": [asdict(m) for m in g_matches],
            "graph_match_count": len(g_matches),
        }

    # ── MCP Prompts (slash-command shortcuts) ──

    @mcp.prompt()
    def cb(question: str, root: str = ".") -> str:
        """Quick context-compressed answer about your codebase. Usage: /cb <question>"""
        return (
            f"The user wants a context-compressed answer about their codebase at '{root}'.\n\n"
            f"Question: {question}\n\n"
            "Instructions:\n"
            "1. Call the search_and_compress tool with user_prompt set to the question above "
            f"and root=\"{root}\". This searches the codebase and compresses the results.\n"
            "2. Use the compressed prompt from the result to answer the question.\n"
            "3. Cite file paths from the kb_matches in your answer."
        )

    @mcp.prompt()
    def cb_deep(question: str, root: str = ".") -> str:
        """Deep codebase search using vector + graph indexes. Usage: /cb_deep <question>"""
        return (
            f"The user wants a thorough, index-backed answer about their codebase at '{root}'.\n\n"
            f"Question: {question}\n\n"
            "Instructions:\n"
            "1. Call vector_graph_search_and_compress with user_prompt set to the question "
            f"above and root=\"{root}\". This uses semantic + graph search for best results.\n"
            "2. If the tool fails (index not built), fall back to search_and_compress instead.\n"
            "3. Use the compressed prompt from the result to answer the question.\n"
            "4. Cite file paths and function names from the matches in your answer."
        )

    @mcp.prompt()
    def cb_index(root: str = ".") -> str:
        """Build both vector and graph indexes for a repo. Usage: /cb_index"""
        return (
            f"The user wants to set up ContextBuddy indexes for the repo at '{root}'.\n\n"
            "Instructions:\n"
            f"1. Call vector_build with root=\"{root}\" to create the semantic index.\n"
            f"2. Call graph_build with root=\"{root}\" to create the code graph index.\n"
            "3. Report the stats from both builds.\n"
            "4. Tell the user they can now use /cb_deep for best-quality searches."
        )

    return mcp


def main() -> None:
    """
    Entry point for `contextbuddy-mcp`.
    """
    mcp = create_server()
    # stdio transport by default (works with MCP Inspector / Claude Desktop / Cursor)
    mcp.run()

