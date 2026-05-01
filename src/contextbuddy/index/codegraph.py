from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .graph import DEFAULT_GRAPH_EXCLUDE_DIRS, DEFAULT_GRAPH_EXTS, _file_fingerprint, _iter_files, _now_iso, _safe_read_text, _sha1


def _require_tree_sitter() -> Tuple[Any, Optional[Any], Optional[Any]]:
    """
    Optional dependency loader.

    Install with: pip install "contextbuddy[codegraph]"
    """
    try:
        from tree_sitter import Parser  # type: ignore
        from tree_sitter_languages import get_language  # type: ignore
        try:
            from tree_sitter_languages import get_parser  # type: ignore
        except Exception:
            get_parser = None  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "Codegraph indexing requires optional dependencies. "
            "Install with: pip install \"contextbuddy[codegraph]\""
        ) from e
    return Parser, get_language, get_parser


def _make_python_parser() -> Any:
    """
    Build a Python parser across tree_sitter_languages versions.

    Some versions expose `get_parser("python")` (preferred).
    Others expose `get_language("python")` + you attach it to Parser.
    """
    Parser, get_language, get_parser = _require_tree_sitter()
    if get_parser is not None:
        try:
            return get_parser("python")
        except Exception:
            # Fall back to language API
            pass
    parser = Parser()
    lang = get_language("python")
    _set_parser_language(parser, lang)
    return parser


def _set_parser_language(parser: Any, language: Any) -> None:
    """
    tree_sitter API compatibility:
    - older: Parser.set_language(Language)
    - newer: parser.language = Language
    """
    if hasattr(parser, "set_language"):
        parser.set_language(language)
        return
    setattr(parser, "language", language)


@dataclass(frozen=True)
class CodeGraphEdge:
    kind: str  # "calls"
    caller: str
    callee: str
    path: str  # repo-relative file path
    line: int = 0
    col: int = 0


class RepoCodeGraphIndex:
    """
    Optional, higher-accuracy Python code relationship index using tree-sitter.

    Current scope (v1):
    - Extract function/class definitions (for qualification) and call edges.
    - Persist edges to a small JSONL file for MCP usage.
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
        self.exts = list(exts) if exts is not None else [".py"]

        self._index_path = self.index_dir / _sha1(str(self.root))
        self._manifest_path = self._index_path / "codegraph_manifest.json"
        self._edges_path = self._index_path / "codegraph_edges.jsonl"

        self._manifest: Dict[str, Any] = {}

    def exists(self) -> bool:
        return self._manifest_path.exists() and self._edges_path.exists()

    def load_manifest(self) -> Dict[str, Any]:
        if not self.exists():
            raise FileNotFoundError(f"Codegraph index not found at {self._index_path}")
        self._manifest = json.loads(self._manifest_path.read_text(encoding="utf-8"))
        return dict(self._manifest)

    def build(
        self,
        *,
        max_files: int = 50_000,
        max_bytes_per_file: int = 512_000,
    ) -> Dict[str, Any]:
        if not self.root.exists() or not self.root.is_dir():
            raise FileNotFoundError(f"Repo root not found or not a directory: {self.root}")
        self._index_path.mkdir(parents=True, exist_ok=True)

        parser = _make_python_parser()

        fingerprints: Dict[str, Dict[str, Any]] = {}
        edge_count = 0
        file_count = 0

        tmp_edges = self._edges_path.with_suffix(".jsonl.tmp")
        with tmp_edges.open("w", encoding="utf-8") as out:
            for fp in _iter_files(self.root, exts=self.exts, exclude_dirs=self.exclude_dirs, max_files=int(max_files)):
                if fp.suffix.lower() != ".py":
                    continue
                file_count += 1
                rel = str(fp.relative_to(self.root))
                fingerprints[rel] = _file_fingerprint(fp)
                text = _safe_read_text(fp, max_bytes=int(max_bytes_per_file))
                if text is None:
                    continue
                edges = _extract_python_calls(parser, text, rel)
                for e in edges:
                    out.write(json.dumps(asdict(e), ensure_ascii=False) + "\n")
                edge_count += len(edges)

        tmp_edges.replace(self._edges_path)
        self._manifest = {
            "schema_version": self.SCHEMA_VERSION,
            "root": str(self.root),
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
            "exclude_dirs": list(self.exclude_dirs),
            "exts": list(self.exts),
            "files": fingerprints,
            "stats": {"files_seen": int(file_count), "call_edges": int(edge_count)},
        }
        self._manifest_path.write_text(json.dumps(self._manifest, ensure_ascii=False), encoding="utf-8")
        return {"index_path": str(self._index_path), **self._manifest.get("stats", {}), "updated_at": self._manifest["updated_at"]}

    def update(
        self,
        *,
        max_files: int = 50_000,
        max_bytes_per_file: int = 512_000,
        prune_deleted: bool = True,
    ) -> Dict[str, Any]:
        if not self.exists():
            return self.build(max_files=max_files, max_bytes_per_file=max_bytes_per_file)
        self.load_manifest()

        old_files: Dict[str, Dict[str, Any]] = dict(self._manifest.get("files", {}))
        new_files: Dict[str, Dict[str, Any]] = {}
        changed: Set[str] = set()
        seen: Set[str] = set()

        for fp in _iter_files(self.root, exts=self.exts, exclude_dirs=self.exclude_dirs, max_files=int(max_files)):
            if fp.suffix.lower() != ".py":
                continue
            rel = str(fp.relative_to(self.root))
            seen.add(rel)
            new_fp = _file_fingerprint(fp)
            new_files[rel] = new_fp
            if old_files.get(rel) != new_fp:
                changed.add(rel)

        deleted = set(old_files.keys()) - seen
        if not prune_deleted:
            deleted = set()

        # Load existing edges and keep unaffected
        keep_paths = set(old_files.keys()) - changed - deleted
        kept: List[Dict[str, Any]] = []
        with self._edges_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if obj.get("path") in keep_paths:
                    kept.append(obj)

        parser = _make_python_parser()

        added_edges = 0
        new_edge_objs: List[Dict[str, Any]] = []
        for rel in sorted(changed):
            abs_fp = (self.root / rel).resolve()
            text = _safe_read_text(abs_fp, max_bytes=int(max_bytes_per_file))
            if text is None:
                continue
            edges = _extract_python_calls(parser, text, rel)
            for e in edges:
                new_edge_objs.append(asdict(e))
            added_edges += len(edges)

        combined = [*kept, *new_edge_objs]
        tmp_edges = self._edges_path.with_suffix(".jsonl.tmp")
        with tmp_edges.open("w", encoding="utf-8") as out:
            for obj in combined:
                out.write(json.dumps(obj, ensure_ascii=False) + "\n")
        tmp_edges.replace(self._edges_path)

        self._manifest["updated_at"] = _now_iso()
        self._manifest["files"] = new_files
        self._manifest["stats"] = {"files_seen": int(len(new_files)), "call_edges": int(len(combined))}
        self._manifest_path.write_text(json.dumps(self._manifest, ensure_ascii=False), encoding="utf-8")

        return {
            "index_path": str(self._index_path),
            "files_scanned": int(len(new_files)),
            "files_changed": int(len(changed)),
            "files_deleted": int(len(deleted)),
            "edges_added": int(added_edges),
            "edges_total": int(len(combined)),
            "updated_at": self._manifest["updated_at"],
        }

    def top_calls_for_paths(self, rel_paths: Sequence[str], *, limit: int = 30) -> List[CodeGraphEdge]:
        """
        Return call edges for the given repo-relative file paths.
        """
        want = {str(p).replace("\\", "/") for p in rel_paths if p}
        out: List[CodeGraphEdge] = []
        if not want:
            return out
        if not self.exists():
            raise FileNotFoundError(f"Codegraph index not found at {self._index_path}")
        with self._edges_path.open("r", encoding="utf-8") as f:
            for line in f:
                if len(out) >= int(limit):
                    break
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                p = str(obj.get("path") or "").replace("\\", "/")
                if p in want:
                    out.append(CodeGraphEdge(**obj))
        return out


def _node_text(src: bytes, node: Any) -> str:
    try:
        return src[node.start_byte : node.end_byte].decode("utf-8", errors="replace")
    except Exception:
        return ""


def _extract_python_calls(parser: Any, text: str, rel_path: str) -> List[CodeGraphEdge]:
    src = text.encode("utf-8", errors="replace")
    tree = parser.parse(src)
    root = tree.root_node

    edges: List[CodeGraphEdge] = []
    scope: List[str] = []

    def qualname() -> str:
        return ".".join(scope) if scope else "<module>"

    def walk(node: Any) -> None:
        nonlocal edges
        t = getattr(node, "type", "")

        if t in ("function_definition", "class_definition"):
            # child_by_field_name("name") exists for these nodes
            nm = node.child_by_field_name("name")
            name = _node_text(src, nm).strip() if nm is not None else ""
            if name:
                scope.append(name)
            # Walk body only
            body = node.child_by_field_name("body")
            if body is not None:
                walk(body)
            if name:
                scope.pop()
            return

        if t == "call":
            fn = node.child_by_field_name("function")
            callee = _node_text(src, fn).strip() if fn is not None else ""
            if callee:
                pt = getattr(node, "start_point", (0, 0))
                edges.append(
                    CodeGraphEdge(
                        kind="calls",
                        caller=qualname(),
                        callee=callee,
                        path=rel_path.replace("\\", "/"),
                        line=int(pt[0]) + 1,
                        col=int(pt[1]) + 1,
                    )
                )

        for ch in getattr(node, "children", []) or []:
            walk(ch)

    walk(root)
    return edges

