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
from typing import Any, Dict, List, Optional, Sequence, Tuple


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
    from ..retriever import rrf_fuse

    try:  # optional
        from ..index.codegraph import RepoCodeGraphIndex
    except Exception:  # pragma: no cover
        RepoCodeGraphIndex = None  # type: ignore

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

    def _git_head(rootp: Path) -> Optional[str]:
        try:
            if not (rootp / ".git").exists():
                return None
            import subprocess

            res = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=str(rootp),
                capture_output=True,
                text=True,
                timeout=2.0,
                check=False,
            )
            head = (res.stdout or "").strip()
            return head or None
        except Exception:
            return None

    def _ensure_repo_index_dir(rootp: Path) -> Path:
        """
        Store indexes inside the repo under `.contextbuddy/indexes` and ensure it is gitignored.
        """
        home = (rootp / ".contextbuddy" / "indexes").resolve()
        try:
            home.mkdir(parents=True, exist_ok=True)
        except Exception:
            return build_default_index_dir()

        try:
            gi = (rootp / ".gitignore").resolve()
            entry = ".contextbuddy/"
            if gi.exists() and gi.is_file():
                existing = gi.read_text(encoding="utf-8", errors="replace")
                if entry not in existing:
                    suffix = "" if existing.endswith("\n") or existing == "" else "\n"
                    gi.write_text(existing + suffix + entry + "\n", encoding="utf-8")
        except Exception:
            pass
        return home

    def _overview_manifest_context(rootp: Path, *, max_files: int = 20, max_depth: int = 2) -> List[str]:
        """
        Include high-signal "how to run / what is it" files (README, manifests, schemas).

        Works for monorepos by scanning a shallow directory depth.
        """
        want_names = {
            "readme.md",
            "readme.txt",
            "pyproject.toml",
            "requirements.txt",
            "setup.py",
            "setup.cfg",
            "package.json",
            "cargo.toml",
            "go.mod",
            "dockerfile",
            "docker-compose.yml",
            "docker-compose.yaml",
            ".env.example",
        }
        out: List[str] = []

        def add_file(p: Path) -> None:
            nonlocal out
            if len(out) >= int(max_files):
                return
            txt = _read_small_text_file(p)
            if not txt:
                return
            out.append(f"Source: {p}\n{txt}".strip())

        # Always try root-level first
        for nm in [
            "README.md",
            "pyproject.toml",
            "requirements.txt",
            "package.json",
            "prisma/schema.prisma",
        ]:
            add_file((rootp / nm).resolve())

        # Shallow walk for common subproject manifests
        root_depth = len(rootp.parts)
        for dirpath, dirnames, filenames in os.walk(rootp):
            if len(out) >= int(max_files):
                break
            depth = len(Path(dirpath).parts) - root_depth
            if depth > int(max_depth):
                dirnames[:] = []
                continue
            # prune common noise dirs quickly
            dirnames[:] = [d for d in dirnames if d not in {".git", ".venv", "node_modules", ".next", "dist", "build", ".contextbuddy"}]

            for fn in filenames:
                if len(out) >= int(max_files):
                    break
                low = fn.lower()
                if low == "readme.md" or low in want_names:
                    add_file((Path(dirpath) / fn).resolve())
                if low == "schema.prisma" and Path(dirpath).name.lower() == "prisma":
                    add_file((Path(dirpath) / fn).resolve())

        # Deduplicate
        seen = set()
        deduped: List[str] = []
        for ch in out:
            k = ch.strip()
            if not k or k in seen:
                continue
            seen.add(k)
            deduped.append(k)
        return deduped

    def _rel_from_abs(rootp: Path, p: str) -> Optional[str]:
        try:
            return str(Path(p).resolve().relative_to(rootp.resolve())).replace("\\", "/")
        except Exception:
            return None

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
        graph_hop_limit: int = 1,
        include_imports: bool = True,
        include_importers: bool = False,
        max_preview_chars: int = 900,
        max_preview_lines: int = 120,
        include_manifest_files: bool = True,
        manifest_max_files: int = 20,
        max_context_tokens: int = 2000,
        min_relevance: float = 0.15,
        conservative_mode: bool = False,
        include_entities_section: bool = True,
        include_structured: bool = True,
        include_key_flows: bool = True,
        key_flow_limit: int = 25,
        auto_build_graph: bool = True,
        graph_build_max_files: int = 50_000,
        graph_build_max_bytes_per_file: int = 512_000,
        auto_build_codegraph: bool = False,
        codegraph_build_max_files: int = 50_000,
        codegraph_build_max_bytes_per_file: int = 512_000,
        auto_update_indexes: bool = True,
    ) -> Dict[str, Any]:
        """
        UX-friendly repo overview: retrieve + compress with minimal knobs.

        Retrieval strategy:
        - If vector and/or graph indexes exist, use them.
        - Fuse vector+graph file rankings via RRF for better coverage.
        - Always (optionally) include README/manifests/schemas for "how to run" answers.
        - Optionally include Python key flows if the optional codegraph index exists.
        """
        rootp = validate_root(root)
        idx_dir = Path(index_dir).resolve() if index_dir else _ensure_repo_index_dir(rootp)

        q = (
            str(overview_query)
            if overview_query
            else "features overview capabilities what does this project do quickstart install run usage "
            "cli commands api routes architecture components entrypoints"
        )

        # indexes
        v = RepoVectorIndex(root=rootp, index_dir=idx_dir, embedder_id=str(embedder_id), embedder_config=dict(embedder_config or {}))
        g = RepoGraphIndex(root=rootp, index_dir=idx_dir)

        if bool(auto_build_graph) and not g.exists():
            g.build(max_files=int(graph_build_max_files), max_bytes_per_file=int(graph_build_max_bytes_per_file))

        v_ok = v.exists()
        g_ok = g.exists()

        # staleness check (git head mismatch) + optional auto-update
        current_head = _git_head(rootp)
        graph_stale = False
        vector_stale = False
        try:
            if g_ok:
                g.load()
                gh = (g._manifest or {}).get("git_head")  # type: ignore[attr-defined]
                graph_stale = bool(current_head and gh and gh != current_head)
        except Exception:
            pass
        try:
            if v_ok:
                v.load()
                vh = (v._data or {}).get("git_head")  # type: ignore[attr-defined]
                vector_stale = bool(current_head and vh and vh != current_head)
        except Exception:
            pass

        if bool(auto_update_indexes):
            if g_ok and graph_stale:
                try:
                    g.update(max_files=int(graph_build_max_files), max_bytes_per_file=int(graph_build_max_bytes_per_file))
                    g_ok = g.exists()
                    graph_stale = False
                except Exception:
                    pass
            if v_ok and vector_stale:
                try:
                    v.update(max_files=int(graph_build_max_files), max_bytes_per_file=int(graph_build_max_bytes_per_file))
                    v_ok = v.exists()
                    vector_stale = False
                except Exception:
                    pass

        v_matches: List[Any] = []
        g_matches: List[Any] = []
        mode = "fallback"

        if v_ok:
            v_matches = v.search(
                query=str(q),
                top_k=int(vector_top_k),
                min_score=float(vector_min_score),
                max_preview_chars=int(max_preview_chars),
            )
        if g_ok:
            g_matches = g.search(
                query=str(q),
                top_k=int(vector_top_k),
                hop_limit=int(graph_hop_limit),
                include_imports=bool(include_imports),
                include_importers=bool(include_importers),
                max_preview_lines=int(max_preview_lines),
            )

        # Decide mode label
        if v_ok and g_ok:
            mode = "vector_graph"
        elif v_ok:
            mode = "vector"
        elif g_ok:
            mode = "graph"

        # Rank fusion for file coverage
        v_rel = [(_rel_from_abs(rootp, m.path) or "") for m in v_matches]
        g_rel = [(_rel_from_abs(rootp, m.path) or "") for m in g_matches]
        fused = rrf_fuse([ [p for p in v_rel if p], [p for p in g_rel if p] ])
        top_files = [p for p, _ in fused[: max(10, int(vector_top_k))] if p] or [p for p in g_rel if p][: max(10, int(vector_top_k))]
        top_set = set(top_files)

        v_context = v.matches_to_context([m for m in v_matches if (_rel_from_abs(rootp, m.path) or "") in top_set]) if v_ok else []
        # graph: expand_from_files using fused top files if available
        g_context: List[str] = []
        g_expanded: List[Any] = []
        if g_ok:
            seed_abs = [str((rootp / rel).resolve()) for rel in top_files[: max(5, int(graph_hop_limit) * 10)]]
            g_expanded = g.expand_from_files(
                seed_abs,
                hop_limit=int(graph_hop_limit),
                include_imports=bool(include_imports),
                include_importers=bool(include_importers),
                top_k=max(10, int(vector_top_k) * 2),
                max_preview_lines=int(max_preview_lines),
            )
            g_context = g.matches_to_context(g_expanded)
        else:
            g_expanded = g_matches

        context_chunks: List[str] = []
        for ch in [*v_context, *g_context]:
            key = ch.strip()
            if not key:
                continue
            context_chunks.append(ch)

        if bool(include_manifest_files):
            context_chunks.extend(_overview_manifest_context(rootp, max_files=int(manifest_max_files)))

        # dedupe
        seen = set()
        deduped: List[str] = []
        for ch in context_chunks:
            k = (ch or "").strip()
            if not k or k in seen:
                continue
            seen.add(k)
            deduped.append(k)

        # structured extraction fields
        structured: Dict[str, Any] = {}
        if bool(include_structured):
            rels = [r for r in top_files if r]
            # entrypoints: simple heuristics
            ep: List[str] = []
            for r in rels:
                low = Path(r).name.lower()
                if low in {"__main__.py", "main.py", "cli.py", "app.py", "server.py", "manage.py"}:
                    ep.append(r)
            structured["entrypoints"] = sorted(set(ep))[:20]

            # core modules: top-level dirs from fused files
            mods: Dict[str, int] = {}
            for r in rels:
                parts = r.split("/")
                if not parts:
                    continue
                top = parts[0]
                mods[top] = mods.get(top, 0) + 1
            structured["core_modules"] = sorted(mods.items(), key=lambda kv: kv[1], reverse=True)[:20]

            structured["index_status"] = {
                "mode": mode,
                "index_dir": str(idx_dir),
                "vector_index_exists": bool(v_ok),
                "graph_index_exists": bool(g_ok),
                "git_head": current_head,
                "graph_index_stale": bool(graph_stale),
                "vector_index_stale": bool(vector_stale),
            }

            # key flows: Python call edges (optional index)
            if bool(include_key_flows) and RepoCodeGraphIndex is not None:
                try:
                    cg = RepoCodeGraphIndex(root=rootp, index_dir=idx_dir)
                    if bool(auto_build_codegraph) and not cg.exists():
                        cg.build(max_files=int(codegraph_build_max_files), max_bytes_per_file=int(codegraph_build_max_bytes_per_file))
                    if cg.exists():
                        edges = cg.top_calls_for_paths(rels, limit=int(key_flow_limit))
                        structured["key_flows"] = [
                            f"{e.caller} -> {e.callee} ({e.path}:{e.line})" for e in edges if e.caller and e.callee and e.path
                        ]
                    else:
                        structured["key_flows"] = []
                except Exception:
                    structured["key_flows"] = []

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

        resp: Dict[str, Any] = {
            "prompt": prompt,
            "report": asdict(report),
            "mode": mode,
            "vector_matches": [asdict(m) for m in v_matches],
            "vector_match_count": len(v_matches),
            "graph_matches": [asdict(m) for m in g_expanded],
            "graph_match_count": len(g_expanded),
        }
        if bool(include_structured):
            resp.update(structured)
        return resp

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

