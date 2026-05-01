from __future__ import annotations

import ast
import json
import os
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


DEFAULT_GRAPH_EXCLUDE_DIRS: Set[str] = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "__pycache__",
    "node_modules",
    "dist",
    "build",
    ".pytest_cache",
}

DEFAULT_GRAPH_EXTS: Set[str] = {".py", ".md", ".txt", ".toml", ".yaml", ".yml", ".json", ".ts", ".tsx", ".js", ".jsx"}


def build_default_index_dir(app_name: str = "ContextBuddy") -> Path:
    """
    Default index home.

    Windows: %LOCALAPPDATA%\\ContextBuddy\\indexes
    Others:  ~/.contextbuddy/indexes
    """
    local = os.environ.get("LOCALAPPDATA")
    if local:
        return Path(local) / app_name / "indexes"
    return Path.home() / f".{app_name.lower()}" / "indexes"


def _sha1(s: str) -> str:
    import hashlib

    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _iter_files(
    root: Path,
    *,
    exts: Sequence[str],
    exclude_dirs: Sequence[str],
    max_files: int,
) -> Iterable[Path]:
    exts_set = {e.lower() for e in exts}
    exclude = set(exclude_dirs)
    n = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in exclude]
        for fn in filenames:
            p = Path(dirpath) / fn
            if exts_set and p.suffix.lower() not in exts_set:
                continue
            yield p
            n += 1
            if max_files > 0 and n >= max_files:
                return


def _file_fingerprint(p: Path) -> Dict[str, Any]:
    st = p.stat()
    return {"mtime_ns": int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))), "size": int(st.st_size)}


def _safe_read_text(p: Path, *, max_bytes: int) -> Optional[str]:
    try:
        if p.stat().st_size > int(max_bytes):
            return None
        text = p.read_text(encoding="utf-8", errors="replace")
        # binary-ish
        if "\x00" in text:
            return None
        return text
    except Exception:
        return None


def _resolve_python_module(root: Path, module: str) -> Optional[str]:
    mod_path = Path(*module.split("."))
    cand1 = (root / mod_path).with_suffix(".py")
    if cand1.exists():
        return str(cand1)
    cand2 = root / mod_path / "__init__.py"
    if cand2.exists():
        return str(cand2)
    return None


def _resolve_python_from_import(file_path: Path, root: Path, module: Optional[str], level: int) -> Optional[str]:
    """
    Resolve `from ...module import ...` into a file path if possible.
    Best-effort only.
    """
    base = file_path.parent
    if level and level > 0:
        # level=1 means "from . import x" (stay in package)
        for _ in range(level - 1):
            base = base.parent
    if module:
        rel = Path(*module.split("."))
        cand1 = (base / rel).with_suffix(".py")
        if cand1.exists():
            return str(cand1)
        cand2 = base / rel / "__init__.py"
        if cand2.exists():
            return str(cand2)
    return None


_js_import_re = re.compile(r"""^\s*import[\s\w{},*\n\r]+from\s+["'](?P<spec>[^"']+)["']""", re.MULTILINE)


def _resolve_js_import(file_path: Path, spec: str) -> Optional[str]:
    if not spec.startswith("."):
        return None
    base = file_path.parent
    cand = (base / spec).resolve()
    # try common extensions
    for ext in (".ts", ".tsx", ".js", ".jsx"):
        p = cand.with_suffix(ext)
        if p.exists():
            return str(p)
    # index files
    for ext in (".ts", ".tsx", ".js", ".jsx"):
        p = cand / ("index" + ext)
        if p.exists():
            return str(p)
    return None


@dataclass(frozen=True)
class GraphSymbol:
    id: str
    name: str
    kind: str  # function|class
    path: str
    start_line: int
    end_line: int


@dataclass(frozen=True)
class GraphMatch:
    kind: str  # symbol|file
    path: str
    score: float
    name: str = ""
    start_line: int = 0
    end_line: int = 0
    preview: str = ""


class RepoGraphIndex:
    """
    Persistent repo graph index (stdlib-only).

    v1 capabilities:
    - File->file edges via import resolution (best-effort) for Python and JS/TS.
    - Python symbols (function/class) with line spans via `ast`.
    - Incremental update by file fingerprints.
    - Query-time retrieval that seeds from symbol/file name overlap and expands 1-2 hops.
    """

    SCHEMA_VERSION = 1

    def __init__(
        self,
        *,
        root: str | Path,
        index_dir: str | Path,
        exclude_dirs: Optional[Sequence[str]] = None,
        exts: Optional[Sequence[str]] = None,
    ) -> None:
        self.root = Path(root).resolve()
        self.index_dir = Path(index_dir).resolve()
        self.exclude_dirs = list(exclude_dirs) if exclude_dirs is not None else sorted(DEFAULT_GRAPH_EXCLUDE_DIRS)
        self.exts = list(exts) if exts is not None else sorted(DEFAULT_GRAPH_EXTS)

        self._index_path = self.index_dir / _sha1(str(self.root))
        self._manifest_path = self._index_path / "manifest.json"
        self._edges_path = self._index_path / "edges.json"
        self._symbols_path = self._index_path / "symbols.jsonl"

        self._manifest: Dict[str, Any] = {}
        self._edges: Dict[str, List[str]] = {}
        self._symbols_by_path: Dict[str, List[GraphSymbol]] = {}

    def exists(self) -> bool:
        return self._manifest_path.exists() and self._edges_path.exists() and self._symbols_path.exists()

    def load(self) -> None:
        if not self.exists():
            raise FileNotFoundError(f"Graph index not found at {self._index_path}")
        self._manifest = json.loads(self._manifest_path.read_text(encoding="utf-8"))
        self._edges = json.loads(self._edges_path.read_text(encoding="utf-8"))
        self._symbols_by_path = {}
        with self._symbols_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                sym = GraphSymbol(**obj)
                self._symbols_by_path.setdefault(sym.path, []).append(sym)

    def build(self, *, max_files: int = 50_000, max_bytes_per_file: int = 512_000) -> Dict[str, Any]:
        if not self.root.exists() or not self.root.is_dir():
            raise FileNotFoundError(f"Repo root not found or not a directory: {self.root}")
        self._index_path.mkdir(parents=True, exist_ok=True)

        edges: Dict[str, Set[str]] = {}
        symbols_by_path: Dict[str, List[GraphSymbol]] = {}
        fingerprints: Dict[str, Dict[str, Any]] = {}

        file_count = 0
        sym_count = 0
        edge_count = 0

        for fp in _iter_files(self.root, exts=self.exts, exclude_dirs=self.exclude_dirs, max_files=int(max_files)):
            file_count += 1
            rel = str(fp.relative_to(self.root))
            fingerprints[rel] = _file_fingerprint(fp)
            text = _safe_read_text(fp, max_bytes=int(max_bytes_per_file))
            if text is None:
                continue

            outs: Set[str] = set()
            if fp.suffix.lower() == ".py":
                file_edges, file_syms = self._parse_python(fp, text)
                outs |= file_edges
                if file_syms:
                    symbols_by_path[rel] = file_syms
                    sym_count += len(file_syms)
            elif fp.suffix.lower() in {".ts", ".tsx", ".js", ".jsx"}:
                file_edges = self._parse_js(fp, text)
                outs |= file_edges
            # other files: no edges/symbols in v1

            if outs:
                edges[rel] = outs
                edge_count += len(outs)

        self._edges = {k: sorted(v) for k, v in edges.items()}
        self._symbols_by_path = symbols_by_path
        self._manifest = {
            "schema_version": self.SCHEMA_VERSION,
            "root": str(self.root),
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
            "exclude_dirs": list(self.exclude_dirs),
            "exts": list(self.exts),
            "files": fingerprints,
            "stats": {"files_seen": file_count, "symbols": sym_count, "edges": edge_count},
        }
        self._save()
        return {"index_path": str(self._index_path), **self._manifest.get("stats", {}), "updated_at": self._manifest["updated_at"]}

    def update(self, *, max_files: int = 50_000, max_bytes_per_file: int = 512_000, prune_deleted: bool = True) -> Dict[str, Any]:
        if not self.exists():
            return self.build(max_files=max_files, max_bytes_per_file=max_bytes_per_file)
        self.load()

        old_files: Dict[str, Dict[str, Any]] = dict(self._manifest.get("files", {}))
        new_files: Dict[str, Dict[str, Any]] = {}

        changed: Set[str] = set()
        seen: Set[str] = set()

        for fp in _iter_files(self.root, exts=self.exts, exclude_dirs=self.exclude_dirs, max_files=int(max_files)):
            rel = str(fp.relative_to(self.root))
            seen.add(rel)
            new_fp = _file_fingerprint(fp)
            new_files[rel] = new_fp
            old_fp = old_files.get(rel)
            if old_fp != new_fp:
                changed.add(rel)

        deleted = set(old_files.keys()) - seen
        if not prune_deleted:
            deleted = set()

        # remove deleted file entries
        for rel in deleted:
            self._edges.pop(rel, None)
            self._symbols_by_path.pop(rel, None)

        # re-parse changed files
        for rel in changed:
            abs_fp = (self.root / rel).resolve()
            text = _safe_read_text(abs_fp, max_bytes=int(max_bytes_per_file))
            self._edges.pop(rel, None)
            self._symbols_by_path.pop(rel, None)
            if text is None:
                continue

            outs: Set[str] = set()
            if abs_fp.suffix.lower() == ".py":
                file_edges, file_syms = self._parse_python(abs_fp, text)
                outs |= file_edges
                if file_syms:
                    self._symbols_by_path[rel] = file_syms
            elif abs_fp.suffix.lower() in {".ts", ".tsx", ".js", ".jsx"}:
                outs |= self._parse_js(abs_fp, text)
            if outs:
                self._edges[rel] = sorted(outs)

        # persist updated manifest + stores
        self._manifest["updated_at"] = _now_iso()
        self._manifest["files"] = new_files
        self._manifest["stats"] = {
            "files_seen": len(new_files),
            "symbols": sum(len(v) for v in self._symbols_by_path.values()),
            "edges": sum(len(v) for v in self._edges.values()),
        }
        self._save()

        return {
            "index_path": str(self._index_path),
            "files_scanned": len(new_files),
            "files_changed": len(changed),
            "files_deleted": len(deleted),
            **self._manifest.get("stats", {}),
            "updated_at": self._manifest["updated_at"],
        }

    def search(
        self,
        query: str,
        *,
        top_k: int = 20,
        hop_limit: int = 1,
        include_imports: bool = True,
        include_importers: bool = False,
        max_preview_lines: int = 80,
    ) -> List[GraphMatch]:
        """
        Return ranked symbol/file matches, expanded by graph hops.
        """
        q = (query or "").strip()
        if not q:
            return []
        if not self.exists():
            raise FileNotFoundError(f"Graph index not found at {self._index_path}")
        # lazy load once per process lifetime usage
        if not self._manifest:
            self.load()

        terms = [t for t in re.split(r"[^A-Za-z0-9_./-]+", q.lower()) if t]
        if not terms:
            return []

        seed_scores: Dict[Tuple[str, str], float] = {}

        # Symbol seeds (python)
        for rel, syms in self._symbols_by_path.items():
            for s in syms:
                name_low = s.name.lower()
                score = 0.0
                for t in terms:
                    if t and t in name_low:
                        score += 1.0
                if score > 0:
                    seed_scores[("symbol", s.id)] = max(seed_scores.get(("symbol", s.id), 0.0), score + 1.0)

        # File seeds
        for rel in self._manifest.get("files", {}).keys():
            rel_low = rel.lower()
            score = 0.0
            for t in terms:
                if t and t in rel_low:
                    score += 0.7
            if score > 0:
                seed_scores[("file", rel)] = max(seed_scores.get(("file", rel), 0.0), score)

        # If we got nothing, fall back to "contains any term in symbol OR file"
        if not seed_scores:
            for rel, syms in self._symbols_by_path.items():
                for s in syms:
                    if any(t in s.name.lower() for t in terms):
                        seed_scores[("symbol", s.id)] = 1.0
            for rel in self._manifest.get("files", {}).keys():
                if any(t in rel.lower() for t in terms):
                    seed_scores[("file", rel)] = 0.7

        # pick top seed ids
        seeds = sorted(seed_scores.items(), key=lambda kv: (-float(kv[1]), kv[0][0], kv[0][1]))[: max(5, int(top_k))]

        # map symbol id -> symbol
        sym_index: Dict[str, GraphSymbol] = {}
        for rel, syms in self._symbols_by_path.items():
            for s in syms:
                sym_index[s.id] = s

        selected_files: Dict[str, float] = {}
        selected_syms: Dict[str, float] = {}

        for (kind, id_), sc in seeds:
            if kind == "file":
                selected_files[id_] = max(selected_files.get(id_, 0.0), float(sc))
            else:
                selected_syms[id_] = max(selected_syms.get(id_, 0.0), float(sc))
                s = sym_index.get(id_)
                if s:
                    selected_files[s.path] = max(selected_files.get(s.path, 0.0), float(sc) * 0.5)

        # Expand via imports edges
        if hop_limit > 0 and include_imports:
            frontier = set(selected_files.keys())
            for _ in range(int(hop_limit)):
                nxt: Set[str] = set()
                for rel in frontier:
                    for dep in self._edges.get(rel, []):
                        if dep not in selected_files:
                            selected_files[dep] = max(0.25, selected_files.get(rel, 0.0) * 0.5)
                        nxt.add(dep)
                frontier = nxt

        # Expand importers (reverse edges) if requested (more expensive)
        if hop_limit > 0 and include_importers:
            rev: Dict[str, Set[str]] = {}
            for src, dsts in self._edges.items():
                for d in dsts:
                    rev.setdefault(d, set()).add(src)
            frontier = set(selected_files.keys())
            for _ in range(int(hop_limit)):
                nxt = set()
                for rel in frontier:
                    for imp in rev.get(rel, set()):
                        if imp not in selected_files:
                            selected_files[imp] = max(0.2, selected_files.get(rel, 0.0) * 0.4)
                        nxt.add(imp)
                frontier = nxt

        # Build matches (prefer symbols, then files)
        matches: List[GraphMatch] = []

        for sym_id, sc in sorted(selected_syms.items(), key=lambda kv: -float(kv[1]))[: int(top_k)]:
            s = sym_index.get(sym_id)
            if not s:
                continue
            preview = self._read_span_preview(s.path, s.start_line, s.end_line, max_lines=int(max_preview_lines))
            matches.append(
                GraphMatch(
                    kind="symbol",
                    path=str((self.root / s.path).resolve()),
                    score=float(sc),
                    name=s.name,
                    start_line=int(s.start_line),
                    end_line=int(s.end_line),
                    preview=preview,
                )
            )

        # Fill remainder with file matches
        if len(matches) < int(top_k):
            for rel, sc in sorted(selected_files.items(), key=lambda kv: -float(kv[1])):
                if len(matches) >= int(top_k):
                    break
                preview = self._read_file_preview(rel, query=q, max_lines=int(max_preview_lines))
                matches.append(
                    GraphMatch(
                        kind="file",
                        path=str((self.root / rel).resolve()),
                        score=float(sc),
                        name=rel,
                        preview=preview,
                    )
                )

        return matches[: int(top_k)]

    def matches_to_context(self, matches: Sequence[GraphMatch]) -> List[str]:
        chunks: List[str] = []
        for m in matches:
            if m.kind == "symbol" and m.start_line and m.end_line:
                src = f"{m.path}:{m.start_line}-{m.end_line}"
            else:
                src = m.path
            body = (m.preview or "").strip()
            if body:
                chunks.append(f"Source: {src}\n{body}".strip())
        return chunks

    def expand_from_files(
        self,
        file_paths: Sequence[str],
        *,
        hop_limit: int = 1,
        include_imports: bool = True,
        include_importers: bool = False,
        top_k: int = 30,
        max_preview_lines: int = 80,
    ) -> List[GraphMatch]:
        """
        Expand the graph starting from a set of absolute file paths.

        Used for hybrid retrieval: vector hits seed the relevant files, then the graph
        pulls in dependencies/importers for completeness.
        """
        if not file_paths:
            return []
        if not self.exists():
            raise FileNotFoundError(f"Graph index not found at {self._index_path}")
        if not self._manifest:
            self.load()

        # map abs path -> repo-relative
        seeds: Set[str] = set()
        for p in file_paths:
            try:
                rel = str(Path(p).resolve().relative_to(self.root))
                seeds.add(rel)
            except Exception:
                continue
        if not seeds:
            return []

        selected_files: Dict[str, float] = {rel: 1.0 for rel in seeds}

        if hop_limit > 0 and include_imports:
            frontier = set(seeds)
            for _ in range(int(hop_limit)):
                nxt: Set[str] = set()
                for rel in frontier:
                    for dep in self._edges.get(rel, []):
                        if dep not in selected_files:
                            selected_files[dep] = max(0.25, selected_files.get(rel, 0.0) * 0.5)
                        nxt.add(dep)
                frontier = nxt

        if hop_limit > 0 and include_importers:
            rev: Dict[str, Set[str]] = {}
            for src, dsts in self._edges.items():
                for d in dsts:
                    rev.setdefault(d, set()).add(src)
            frontier = set(seeds)
            for _ in range(int(hop_limit)):
                nxt = set()
                for rel in frontier:
                    for imp in rev.get(rel, set()):
                        if imp not in selected_files:
                            selected_files[imp] = max(0.2, selected_files.get(rel, 0.0) * 0.4)
                        nxt.add(imp)
                frontier = nxt

        matches: List[GraphMatch] = []
        for rel, sc in sorted(selected_files.items(), key=lambda kv: -float(kv[1]))[: int(top_k)]:
            preview = self._read_file_preview(rel, query=rel, max_lines=int(max_preview_lines))
            matches.append(
                GraphMatch(
                    kind="file",
                    path=str((self.root / rel).resolve()),
                    score=float(sc),
                    name=rel,
                    preview=preview,
                )
            )
        return matches

    def _parse_python(self, fp: Path, text: str) -> Tuple[Set[str], List[GraphSymbol]]:
        rel = str(fp.relative_to(self.root))
        edges: Set[str] = set()
        symbols: List[GraphSymbol] = []
        try:
            tree = ast.parse(text)
        except Exception:
            return edges, symbols

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    target = _resolve_python_module(self.root, alias.name)
                    if target:
                        edges.add(str(Path(target).resolve().relative_to(self.root)))
            elif isinstance(node, ast.ImportFrom):
                target = _resolve_python_from_import(fp, self.root, node.module, int(getattr(node, "level", 0) or 0))
                if target:
                    try:
                        edges.add(str(Path(target).resolve().relative_to(self.root)))
                    except Exception:
                        pass
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if not getattr(node, "lineno", None):
                    continue
                start = int(node.lineno)
                end = int(getattr(node, "end_lineno", node.lineno))
                kind = "class" if isinstance(node, ast.ClassDef) else "function"
                name = str(getattr(node, "name", ""))
                if not name:
                    continue
                sid = _sha1(f"{rel}:{name}:{kind}:{start}:{end}")
                symbols.append(
                    GraphSymbol(
                        id=sid,
                        name=name,
                        kind=kind,
                        path=rel,
                        start_line=start,
                        end_line=end,
                    )
                )
        return edges, symbols

    def _parse_js(self, fp: Path, text: str) -> Set[str]:
        edges: Set[str] = set()
        for m in _js_import_re.finditer(text):
            spec = m.group("spec")
            target = _resolve_js_import(fp, spec)
            if target:
                try:
                    edges.add(str(Path(target).resolve().relative_to(self.root)))
                except Exception:
                    pass
        return edges

    def _read_span_preview(self, rel_path: str, start: int, end: int, *, max_lines: int) -> str:
        p = (self.root / rel_path).resolve()
        try:
            lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            return ""
        if start <= 0 or end <= 0:
            return ""
        s = max(1, int(start))
        e = max(s, int(end))
        # clamp and cap
        s = min(s, len(lines))
        e = min(e, len(lines))
        span = lines[s - 1 : e]
        if len(span) > int(max_lines):
            span = span[: int(max_lines)] + ["..."]
        return "\n".join(span).strip()

    def _read_file_preview(self, rel_path: str, *, query: str, max_lines: int) -> str:
        p = (self.root / rel_path).resolve()
        try:
            lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            return ""
        q = (query or "").strip()
        if not q:
            return "\n".join(lines[: int(max_lines)]).strip()
        terms = [t for t in re.split(r"[^A-Za-z0-9_./-]+", q.lower()) if t]
        if not terms:
            return "\n".join(lines[: int(max_lines)]).strip()
        kept: List[str] = []
        for ln in lines:
            low = ln.lower()
            if any(t in low for t in terms):
                kept.append(ln)
            if len(kept) >= int(max_lines):
                break
        if kept:
            return "\n".join(kept).strip()
        return "\n".join(lines[: int(max_lines)]).strip()

    def _save(self) -> None:
        # manifest + edges are small; write atomically
        tmp_manifest = self._manifest_path.with_suffix(".json.tmp")
        tmp_edges = self._edges_path.with_suffix(".json.tmp")
        tmp_symbols = self._symbols_path.with_suffix(".jsonl.tmp")

        tmp_manifest.write_text(json.dumps(self._manifest, indent=2, sort_keys=True), encoding="utf-8")
        tmp_edges.write_text(json.dumps(self._edges, indent=2, sort_keys=True), encoding="utf-8")

        with tmp_symbols.open("w", encoding="utf-8") as f:
            for rel, syms in sorted(self._symbols_by_path.items()):
                for s in syms:
                    f.write(json.dumps(asdict(s), sort_keys=True) + "\n")

        tmp_manifest.replace(self._manifest_path)
        tmp_edges.replace(self._edges_path)
        tmp_symbols.replace(self._symbols_path)

