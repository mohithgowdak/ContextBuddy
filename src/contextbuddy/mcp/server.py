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

import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Optional, Tuple

from .policy import (
    MCPInputError,
    MCPToolTimeout,
    error_payload_compress,
    index_timeout_sec,
    limits_summary,
    run_with_timeout,
    tool_timeout_sec,
    validate_context,
    validate_prompt,
    validate_query_str,
)


def _configure_mcp_logging() -> None:
    """
    The MCP Python SDK logs routine requests at INFO to stderr. Cursor's MCP output
    panel labels stderr as [error], which looks alarming. Default to WARNING; set
    CONTEXTBUDDY_MCP_LOG_LEVEL=INFO for debugging.
    """
    import logging

    raw = (os.environ.get("CONTEXTBUDDY_MCP_LOG_LEVEL") or "WARNING").strip().upper()
    level = getattr(logging, raw, logging.WARNING)
    for name in (
        "mcp",
        "mcp.server",
        "mcp.server.lowlevel",
        "mcp.server.lowlevel.server",
        "mcp.server.fastmcp",
        "mcp.server.session",
        "uvicorn",
        "uvicorn.error",
        "uvicorn.access",
        "httpx",
        "httpcore",
    ):
        logging.getLogger(name).setLevel(level)


def _mcp_vector_prefer_subpaths(raw: Optional[List[str]]) -> List[str]:
    """Default IDE behavior: boost chunks under a top-level `src/` segment unless explicitly disabled."""
    if raw is None:
        return ["src"]
    return list(raw)


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

    # json_response is for Streamable HTTP; keep default for stdio (Cursor / Claude Desktop).
    mcp = FastMCP("ContextBuddy")

    from .. import __version__ as _cb_version

    @mcp.tool()
    def about() -> Dict[str, Any]:
        """
        Server metadata: version, capabilities, and policy limits (no user data logged).
        """
        return {
            "name": "ContextBuddy",
            "version": _cb_version,
            "mcp_tools": [
                "compress",
                "search_kb",
                "search_and_compress",
                "graph_build",
                "graph_update",
                "graph_search",
                "graph_search_and_compress",
                "vector_build",
                "vector_update",
                "vector_search",
                "vector_search_and_compress",
                "vector_graph_search_and_compress",
                "project_overview_and_compress",
                "about",
                "validate_config",
            ],
            "notes": [
                "Does not call an LLM.",
                "User content is not written to server logs (process stdout/stderr).",
                "Input size caps and timeouts are configurable via CONTEXTBUDDY_MCP_* env vars.",
            ],
            "policy": limits_summary(),
        }

    @mcp.tool()
    def validate_config(
        root: Optional[str] = None,
        embedder_id: str = "localhash",
    ) -> Dict[str, Any]:
        """
        Check optional MCP configuration: allowed roots, embedder import, and policy limits.

        Does not read or log repository file contents.
        """
        issues: List[str] = []
        hints: List[str] = []

        if not (os.environ.get("CONTEXTBUDDY_ALLOWED_ROOTS") or "").strip():
            hints.append(
                "Set CONTEXTBUDDY_ALLOWED_ROOTS in production so MCP only searches/indexes approved folders."
            )

        if root:
            try:
                validate_root(root)
            except (FileNotFoundError, PermissionError) as e:
                issues.append(str(e))

        try:
            from ..index.vector import make_embedder

            make_embedder(str(embedder_id), config={})
        except Exception as e:  # pragma: no cover - import errors vary by extras
            issues.append(f"embedder '{embedder_id}': {e}")

        return {
            "ok": len(issues) == 0,
            "issues": issues,
            "hints": hints,
            "limits": limits_summary(),
        }

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
        try:
            validate_prompt(str(user_prompt))
            validate_context(context)
        except MCPInputError as e:
            return error_payload_compress(e.message, e.code)

        def _run() -> Tuple[str, Any]:
            engine = ContextEngine(
                ContextEngineConfig(
                    max_context_tokens=int(max_context_tokens),
                    min_relevance=float(min_relevance),
                    conservative_mode=bool(conservative_mode),
                    dev_mode=False,
                    include_entities_section=bool(include_entities_section),
                )
            )
            return engine.build_prompt(user_prompt=str(user_prompt), context=context)

        try:
            prompt, report = run_with_timeout(tool_timeout_sec(), _run)
        except MCPToolTimeout:
            return error_payload_compress(
                f"Compression exceeded timeout ({tool_timeout_sec()}s). "
                "Increase CONTEXTBUDDY_MCP_TOOL_TIMEOUT_SEC or reduce input size.",
                "TIMEOUT",
            )
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
        try:
            validate_query_str(str(query))
        except MCPInputError as e:
            return {"ok": False, "error": {"code": e.code, "message": e.message}, "matches": [], "count": 0}

        rootp = validate_root(root)

        def _run() -> List[Any]:
            return search_codebase(
                query=str(query),
                root=str(rootp),
                max_matches=int(max_matches),
                max_files=int(max_files),
                max_bytes_per_file=int(max_bytes_per_file),
                context_lines=int(context_lines),
                case_sensitive=bool(case_sensitive),
                group_adjacent=bool(group_adjacent),
            )

        try:
            matches = run_with_timeout(tool_timeout_sec(), _run)
        except MCPToolTimeout:
            return {
                "ok": False,
                "error": {
                    "code": "TIMEOUT",
                    "message": f"search_kb exceeded timeout ({tool_timeout_sec()}s).",
                },
                "matches": [],
                "count": 0,
            }
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
        q = str(kb_query) if kb_query else str(user_prompt)
        try:
            validate_prompt(str(user_prompt))
            validate_query_str(q, field="kb_query")
        except MCPInputError as e:
            return {
                **error_payload_compress(e.message, e.code),
                "kb_matches": [],
                "kb_match_count": 0,
            }

        rootp = validate_root(root)

        def _run() -> Tuple[str, Any, List[Any]]:
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
            return prompt, report, matches

        try:
            prompt, report, matches = run_with_timeout(tool_timeout_sec(), _run)
        except MCPToolTimeout:
            return {
                **error_payload_compress(
                    f"search_and_compress exceeded timeout ({tool_timeout_sec()}s).",
                    "TIMEOUT",
                ),
                "kb_matches": [],
                "kb_match_count": 0,
            }
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

        def _run() -> Dict[str, Any]:
            return idx.build(max_files=int(max_files), max_bytes_per_file=int(max_bytes_per_file))

        try:
            stats = run_with_timeout(index_timeout_sec(), _run)
        except MCPToolTimeout:
            return {
                "ok": False,
                "error": {
                    "code": "TIMEOUT",
                    "message": f"graph_build exceeded timeout ({index_timeout_sec()}s). "
                    "Increase CONTEXTBUDDY_MCP_INDEX_TIMEOUT_SEC if needed.",
                },
                "root": str(rootp),
            }
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

        def _run() -> Dict[str, Any]:
            return idx.update(
                max_files=int(max_files),
                max_bytes_per_file=int(max_bytes_per_file),
                prune_deleted=bool(prune_deleted),
            )

        try:
            stats = run_with_timeout(index_timeout_sec(), _run)
        except MCPToolTimeout:
            return {
                "ok": False,
                "error": {
                    "code": "TIMEOUT",
                    "message": f"graph_update exceeded timeout ({index_timeout_sec()}s).",
                },
                "root": str(rootp),
            }
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
        try:
            validate_query_str(str(query))
        except MCPInputError as e:
            return {"ok": False, "error": {"code": e.code, "message": e.message}, "matches": [], "count": 0}

        rootp = validate_root(root)
        idx_dir = Path(index_dir).resolve() if index_dir else build_default_index_dir()
        idx = RepoGraphIndex(root=rootp, index_dir=idx_dir)

        def _run() -> List[Any]:
            return idx.search(
                query=str(query),
                top_k=int(top_k),
                hop_limit=int(hop_limit),
                include_imports=bool(include_imports),
                include_importers=bool(include_importers),
                max_preview_lines=int(max_preview_lines),
            )

        try:
            matches = run_with_timeout(tool_timeout_sec(), _run)
        except MCPToolTimeout:
            return {
                "ok": False,
                "error": {"code": "TIMEOUT", "message": f"graph_search exceeded timeout ({tool_timeout_sec()}s)."},
                "matches": [],
                "count": 0,
            }
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
        q = str(graph_query) if graph_query else str(user_prompt)
        try:
            validate_prompt(str(user_prompt))
            validate_query_str(q, field="graph_query")
        except MCPInputError as e:
            return {
                **error_payload_compress(e.message, e.code),
                "graph_matches": [],
                "graph_match_count": 0,
            }

        rootp = validate_root(root)
        idx_dir = Path(index_dir).resolve() if index_dir else build_default_index_dir()
        idx = RepoGraphIndex(root=rootp, index_dir=idx_dir)

        def _run() -> Tuple[Any, Any, List[Any]]:
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
            return prompt, report, matches

        try:
            prompt, report, matches = run_with_timeout(tool_timeout_sec(), _run)
        except MCPToolTimeout:
            return {
                **error_payload_compress(
                    f"graph_search_and_compress exceeded timeout ({tool_timeout_sec()}s).",
                    "TIMEOUT",
                ),
                "graph_matches": [],
                "graph_match_count": 0,
            }
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
        exclude_dirs: Optional[List[str]] = None,
        max_files: int = 50_000,
        max_bytes_per_file: int = 512_000,
        batch_size: int = 64,
    ) -> Dict[str, Any]:
        """
        Build a persistent vector index over repo chunks.

        This enables fast semantic-ish search without scanning every file per query.
        `exclude_dirs` is merged with built-in defaults (e.g. `.git`, `node_modules`, `.cursor`).
        """
        rootp = validate_root(root)
        idx_dir = Path(index_dir).resolve() if index_dir else build_default_index_dir()
        idx = RepoVectorIndex(
            root=rootp,
            index_dir=idx_dir,
            embedder_id=str(embedder_id),
            embedder_config=dict(embedder_config or {}),
            exclude_dirs=exclude_dirs,
        )

        def _run() -> Dict[str, Any]:
            return idx.build(
                max_files=int(max_files),
                max_bytes_per_file=int(max_bytes_per_file),
                batch_size=int(batch_size),
            )

        try:
            stats = run_with_timeout(index_timeout_sec(), _run)
        except MCPToolTimeout:
            return {
                "ok": False,
                "error": {
                    "code": "TIMEOUT",
                    "message": f"vector_build exceeded timeout ({index_timeout_sec()}s). "
                    "Increase CONTEXTBUDDY_MCP_INDEX_TIMEOUT_SEC if needed.",
                },
                "root": str(rootp),
            }
        return {"root": str(rootp), **stats}

    @mcp.tool()
    def vector_update(
        root: str = ".",
        index_dir: Optional[str] = None,
        embedder_id: str = "localhash",
        embedder_config: Optional[Dict[str, Any]] = None,
        exclude_dirs: Optional[List[str]] = None,
        max_files: int = 50_000,
        max_bytes_per_file: int = 512_000,
        batch_size: int = 64,
        prune_deleted: bool = True,
    ) -> Dict[str, Any]:
        """
        Incrementally update an existing vector index based on file changes.
        `exclude_dirs` is merged with built-in defaults.
        """
        rootp = validate_root(root)
        idx_dir = Path(index_dir).resolve() if index_dir else build_default_index_dir()
        idx = RepoVectorIndex(
            root=rootp,
            index_dir=idx_dir,
            embedder_id=str(embedder_id),
            embedder_config=dict(embedder_config or {}),
            exclude_dirs=exclude_dirs,
        )

        def _run() -> Dict[str, Any]:
            return idx.update(
                max_files=int(max_files),
                max_bytes_per_file=int(max_bytes_per_file),
                batch_size=int(batch_size),
                prune_deleted=bool(prune_deleted),
            )

        try:
            stats = run_with_timeout(index_timeout_sec(), _run)
        except MCPToolTimeout:
            return {
                "ok": False,
                "error": {"code": "TIMEOUT", "message": f"vector_update exceeded timeout ({index_timeout_sec()}s)."},
                "root": str(rootp),
            }
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
        prefer_subpaths: Optional[List[str]] = None,
        prefer_subpath_boost: float = 1.15,
    ) -> Dict[str, Any]:
        """
        Search the persistent vector index and return ranked chunk matches.
        By default, chunks under a path segment `src` rank higher (set `prefer_subpaths` to `[]` to disable).
        """
        try:
            validate_query_str(str(query))
        except MCPInputError as e:
            return {"ok": False, "error": {"code": e.code, "message": e.message}, "matches": [], "count": 0}

        rootp = validate_root(root)
        idx_dir = Path(index_dir).resolve() if index_dir else build_default_index_dir()
        idx = RepoVectorIndex(
            root=rootp,
            index_dir=idx_dir,
            embedder_id=str(embedder_id),
            embedder_config=dict(embedder_config or {}),
        )

        def _run() -> List[Any]:
            return idx.search(
                query=str(query),
                top_k=int(top_k),
                min_score=float(min_score),
                max_preview_chars=int(max_preview_chars),
                prefer_subpaths=_mcp_vector_prefer_subpaths(prefer_subpaths),
                prefer_subpath_boost=float(prefer_subpath_boost),
            )

        try:
            matches = run_with_timeout(tool_timeout_sec(), _run)
        except MCPToolTimeout:
            return {
                "ok": False,
                "error": {"code": "TIMEOUT", "message": f"vector_search exceeded timeout ({tool_timeout_sec()}s)."},
                "matches": [],
                "count": 0,
            }
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
        prefer_subpaths: Optional[List[str]] = None,
        prefer_subpath_boost: float = 1.15,
        max_context_tokens: int = 2000,
        min_relevance: float = 0.15,
        conservative_mode: bool = False,
        include_entities_section: bool = True,
    ) -> Dict[str, Any]:
        """
        Retrieve context via the vector index, then compress it into a prompt.
        By default, chunks under a path segment `src` rank higher (set `prefer_subpaths` to `[]` to disable).
        """
        q = str(vector_query) if vector_query else str(user_prompt)
        try:
            validate_prompt(str(user_prompt))
            validate_query_str(q, field="vector_query")
        except MCPInputError as e:
            return {
                **error_payload_compress(e.message, e.code),
                "vector_matches": [],
                "vector_match_count": 0,
            }

        rootp = validate_root(root)
        idx_dir = Path(index_dir).resolve() if index_dir else build_default_index_dir()
        idx = RepoVectorIndex(
            root=rootp,
            index_dir=idx_dir,
            embedder_id=str(embedder_id),
            embedder_config=dict(embedder_config or {}),
        )

        def _run() -> Tuple[Any, Any, List[Any]]:
            matches = idx.search(
                query=q,
                top_k=int(top_k),
                min_score=float(min_score),
                max_preview_chars=int(max_preview_chars),
                prefer_subpaths=_mcp_vector_prefer_subpaths(prefer_subpaths),
                prefer_subpath_boost=float(prefer_subpath_boost),
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
            return prompt, report, matches

        try:
            prompt, report, matches = run_with_timeout(tool_timeout_sec(), _run)
        except MCPToolTimeout:
            return {
                **error_payload_compress(
                    f"vector_search_and_compress exceeded timeout ({tool_timeout_sec()}s).",
                    "TIMEOUT",
                ),
                "vector_matches": [],
                "vector_match_count": 0,
            }
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
        vector_prefer_subpaths: Optional[List[str]] = None,
        vector_prefer_subpath_boost: float = 1.15,
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
        Vector seeds prefer paths with a `src` segment by default (`vector_prefer_subpaths=[]` disables).
        """
        q = str(query) if query else str(user_prompt)
        try:
            validate_prompt(str(user_prompt))
            validate_query_str(q, field="query")
        except MCPInputError as e:
            return {
                **error_payload_compress(e.message, e.code),
                "vector_matches": [],
                "vector_match_count": 0,
                "graph_matches": [],
                "graph_match_count": 0,
            }

        rootp = validate_root(root)
        idx_dir = Path(index_dir).resolve() if index_dir else build_default_index_dir()

        def _run() -> Tuple[Any, Any, List[Any], List[Any]]:
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
                prefer_subpaths=_mcp_vector_prefer_subpaths(vector_prefer_subpaths),
                prefer_subpath_boost=float(vector_prefer_subpath_boost),
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
            return prompt, report, v_matches, g_matches

        try:
            prompt, report, v_matches, g_matches = run_with_timeout(tool_timeout_sec(), _run)
        except MCPToolTimeout:
            return {
                **error_payload_compress(
                    f"vector_graph_search_and_compress exceeded timeout ({tool_timeout_sec()}s). "
                    "Increase CONTEXTBUDDY_MCP_TOOL_TIMEOUT_SEC for heavy repos.",
                    "TIMEOUT",
                ),
                "vector_matches": [],
                "vector_match_count": 0,
                "graph_matches": [],
                "graph_match_count": 0,
            }
        return {
            "prompt": prompt,
            "report": asdict(report),
            "vector_matches": [asdict(m) for m in v_matches],
            "vector_match_count": len(v_matches),
            "graph_matches": [asdict(m) for m in g_matches],
            "graph_match_count": len(g_matches),
        }

    def _read_small_text_file(p: Path, *, max_bytes: int = 256_000) -> Optional[str]:
        try:
            if not p.exists() or not p.is_file():
                return None
            if p.stat().st_size <= 0:
                return None
            data = p.read_bytes()
            if len(data) > int(max_bytes):
                data = data[: int(max_bytes)]
            return data.decode("utf-8", errors="replace")
        except Exception:
            return None

    def _overview_fallback_context(rootp: Path, *, max_files: int = 12) -> List[str]:
        """
        No-index fallback: collect a few high-signal project files.
        This is intentionally conservative to avoid blowing context budgets.
        """
        candidates = [
            "README.md",
            "readme.md",
            "README.txt",
            "pyproject.toml",
            "package.json",
            "Cargo.toml",
            "go.mod",
            "requirements.txt",
            "Pipfile",
            "poetry.lock",
            "docker-compose.yml",
            "docker-compose.yaml",
            "Dockerfile",
            ".env.example",
        ]
        out: List[str] = []
        for rel in candidates:
            if len(out) >= int(max_files):
                break
            txt = _read_small_text_file((rootp / rel).resolve())
            if not txt:
                continue
            out.append(f"Source: {(rootp / rel).resolve()}\n{txt}".strip())
        return out

    def _overview_manifest_context(rootp: Path, *, max_files: int = 10) -> List[str]:
        """
        High-signal project metadata files to include even when indexes exist.

        This helps "features/how to run" questions by pulling in run commands,
        scripts, and schemas that graph/vector retrieval may miss.
        """
        candidates = [
            # root-level
            "README.md",
            "readme.md",
            "README.txt",
            "pyproject.toml",
            "package.json",
            "Cargo.toml",
            "go.mod",
            "requirements.txt",
            "Pipfile",
            "poetry.lock",
            "docker-compose.yml",
            "docker-compose.yaml",
            "Dockerfile",
            ".env.example",
            # common monorepo app dir
            "apps/web/README.md",
            "apps/app/README.md",
            "app/README.md",
            # prisma
            "prisma/schema.prisma",
        ]
        out: List[str] = []
        for rel in candidates:
            if len(out) >= int(max_files):
                break
            txt = _read_small_text_file((rootp / rel).resolve())
            if not txt:
                continue
            out.append(f"Source: {(rootp / rel).resolve()}\n{txt}".strip())
        return out

    def _ensure_repo_index_dir(rootp: Path) -> Path:
        """
        Store indexes inside the repo under `.contextbuddy/indexes` and ensure it is gitignored.
        """
        home = (rootp / ".contextbuddy" / "indexes").resolve()
        try:
            home.mkdir(parents=True, exist_ok=True)
        except Exception:
            # If we can't create the folder, fall back to global index dir.
            return build_default_index_dir()

        try:
            gi = (rootp / ".gitignore").resolve()
            entry = ".contextbuddy/"
            if gi.exists() and gi.is_file():
                existing = gi.read_text(encoding="utf-8", errors="replace")
                if entry not in existing:
                    suffix = "" if existing.endswith("\n") or existing == "" else "\n"
                    gi.write_text(existing + suffix + entry + "\n", encoding="utf-8")
            else:
                # If no .gitignore exists, don't create one implicitly.
                pass
        except Exception:
            pass

        return home

    @mcp.tool()
    def project_overview_and_compress(
        user_prompt: str,
        overview_query: Optional[str] = None,
        root: str = ".",
        index_dir: Optional[str] = None,
        embedder_id: str = "localhash",
        embedder_config: Optional[Dict[str, Any]] = None,
        vector_top_k: int = 25,
        vector_min_score: float = 0.0,
        vector_prefer_subpaths: Optional[List[str]] = None,
        vector_prefer_subpath_boost: float = 1.15,
        auto_build_graph: bool = True,
        graph_build_max_files: int = 50_000,
        graph_build_max_bytes_per_file: int = 512_000,
        include_manifest_files: bool = True,
        manifest_max_files: int = 10,
        graph_hop_limit: int = 1,
        include_imports: bool = True,
        include_importers: bool = False,
        max_preview_chars: int = 900,
        max_preview_lines: int = 120,
        max_context_tokens: int = 2000,
        min_relevance: float = 0.15,
        conservative_mode: bool = False,
        include_entities_section: bool = True,
    ) -> Dict[str, Any]:
        """
        UX-friendly "just answer about this repo" tool.

        It automatically chooses the best available retrieval path:
        - vector+graph if both indexes exist
        - vector-only or graph-only if only one exists
        - otherwise, a small fallback context (README / manifests)

        Use this when you don't want users to specify multiple MCP calls.
        """
        q = (
            str(overview_query)
            if overview_query
            else "features overview capabilities what does this project do quickstart install run usage "
            "cli commands mcp tools api endpoints architecture components"
        )
        try:
            validate_prompt(str(user_prompt))
            validate_query_str(str(q), field="overview_query")
        except MCPInputError as e:
            return {
                **error_payload_compress(e.message, e.code),
                "mode": "error",
                "vector_matches": [],
                "vector_match_count": 0,
                "graph_matches": [],
                "graph_match_count": 0,
            }

        rootp = validate_root(root)
        idx_dir = Path(index_dir).resolve() if index_dir else _ensure_repo_index_dir(rootp)

        def _run() -> Tuple[str, Any, str, List[Any], List[Any]]:
            v = RepoVectorIndex(
                root=rootp,
                index_dir=idx_dir,
                embedder_id=str(embedder_id),
                embedder_config=dict(embedder_config or {}),
            )
            g = RepoGraphIndex(root=rootp, index_dir=idx_dir)

            v_ok = v.exists()
            g_ok = g.exists()
            if (not g_ok) and bool(auto_build_graph):
                run_with_timeout(
                    index_timeout_sec(),
                    lambda: g.build(max_files=int(graph_build_max_files), max_bytes_per_file=int(graph_build_max_bytes_per_file)),
                )
                g_ok = g.exists()

            context_chunks: List[str] = []
            v_matches: List[Any] = []
            g_matches: List[Any] = []
            mode = "fallback"

            if v_ok and g_ok:
                mode = "vector_graph"
                v_matches = v.search(
                    query=str(q),
                    top_k=int(vector_top_k),
                    min_score=float(vector_min_score),
                    max_preview_chars=int(max_preview_chars),
                    prefer_subpaths=_mcp_vector_prefer_subpaths(vector_prefer_subpaths),
                    prefer_subpath_boost=float(vector_prefer_subpath_boost),
                )
                v_context = v.matches_to_context(v_matches)
                g_matches = g.expand_from_files(
                    [m.path for m in v_matches],
                    hop_limit=int(graph_hop_limit),
                    include_imports=bool(include_imports),
                    include_importers=bool(include_importers),
                    top_k=max(10, int(vector_top_k) * 2),
                    max_preview_lines=int(max_preview_lines),
                )
                g_context = g.matches_to_context(g_matches)
                context_chunks = [*v_context, *g_context]
            elif v_ok:
                mode = "vector"
                v_matches = v.search(
                    query=str(q),
                    top_k=int(vector_top_k),
                    min_score=float(vector_min_score),
                    max_preview_chars=int(max_preview_chars),
                    prefer_subpaths=_mcp_vector_prefer_subpaths(vector_prefer_subpaths),
                    prefer_subpath_boost=float(vector_prefer_subpath_boost),
                )
                context_chunks = v.matches_to_context(v_matches)
            elif g_ok:
                mode = "graph"
                g_matches = g.search(
                    query=str(q),
                    top_k=int(vector_top_k),
                    hop_limit=int(graph_hop_limit),
                    include_imports=bool(include_imports),
                    include_importers=bool(include_importers),
                    max_preview_lines=int(max_preview_lines),
                )
                context_chunks = g.matches_to_context(g_matches)
            else:
                context_chunks = _overview_fallback_context(rootp)

            if bool(include_manifest_files):
                context_chunks = [*context_chunks, *_overview_manifest_context(rootp, max_files=int(manifest_max_files))]

            # Deduplicate after combining
            seen = set()
            deduped: List[str] = []
            for ch in context_chunks:
                k = (ch or "").strip()
                if not k or k in seen:
                    continue
                seen.add(k)
                deduped.append(k)

            engine = ContextEngine(
                ContextEngineConfig(
                    max_context_tokens=int(max_context_tokens),
                    min_relevance=float(min_relevance),
                    conservative_mode=bool(conservative_mode),
                    dev_mode=False,
                    include_entities_section=bool(include_entities_section),
                )
            )
            prompt, report = engine.build_prompt(user_prompt=str(user_prompt), context=deduped)
            return prompt, report, mode, v_matches, g_matches

        try:
            prompt, report, mode, v_matches, g_matches = run_with_timeout(tool_timeout_sec(), _run)
        except MCPToolTimeout:
            return {
                **error_payload_compress(
                    f"project_overview_and_compress exceeded timeout ({tool_timeout_sec()}s).",
                    "TIMEOUT",
                ),
                "mode": "timeout",
                "vector_matches": [],
                "vector_match_count": 0,
                "graph_matches": [],
                "graph_match_count": 0,
            }

        return {
            "prompt": prompt,
            "report": asdict(report),
            "mode": str(mode),
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
    _configure_mcp_logging()
    mcp = create_server()
    # stdio transport by default (works with MCP Inspector / Claude Desktop / Cursor)
    mcp.run()

