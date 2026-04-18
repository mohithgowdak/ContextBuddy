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
from typing import Any, Dict, Sequence, Optional


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
        """
        Compress raw context into a budgeted prompt (no LLM call).

        Returns:
          - prompt: str
          - report: dict
        """
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
        """
        Search a local codebase/knowledge-base directory for a query and return
        line-based previews with file paths.
        """
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
        """
        Gather initial context by searching a local KB, then compress it into a prompt.

        - kb_query defaults to user_prompt if not provided.
        - This does NOT call an LLM; it returns a compressed prompt + report.
        """
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
        """
        Build a persistent repo graph index (imports + Python symbol spans).

        This is a fast, stdlib-only index meant for IDE usage.
        """
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
        """
        Incrementally update an existing repo graph index based on file changes.
        """
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
        """
        Search the repo graph index and return ranked symbol/file matches.
        """
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
        """
        Retrieve context via the repo graph index, then compress it into a prompt.

        - graph_query defaults to user_prompt if not provided.
        - Returns a compressed prompt + report, plus the graph matches used.
        """
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
        """
        Build a persistent vector index over repo chunks.

        This enables fast semantic-ish search without scanning every file per query.
        """
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
        """
        Incrementally update an existing vector index based on file changes.
        """
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
        """
        Search the persistent vector index and return ranked chunk matches.
        """
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
        """
        Retrieve context via the vector index, then compress it into a prompt.
        """
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
        """
        Best-quality IDE retrieval:
        - Vector search finds the most relevant chunks quickly (seeds).
        - Repo graph expansion pulls in dependencies/importers for completeness.
        - ContextEngine compresses everything into budget.
        """
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

    return mcp


def main() -> None:
    """
    Entry point for `contextbuddy-mcp`.
    """
    mcp = create_server()
    # stdio transport by default (works with MCP Inspector / Claude Desktop / Cursor)
    mcp.run()

