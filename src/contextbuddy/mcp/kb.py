from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_EXCLUDE_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    ".venv-smoke",
    "__pycache__",
    "node_modules",
    "dist",
    "build",
    ".pytest_cache",
}

DEFAULT_EXTS = {
    ".py",
    ".md",
    ".txt",
    ".toml",
    ".yaml",
    ".yml",
    ".json",
    ".ini",
    ".cfg",
    ".rst",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
}


@dataclass(frozen=True)
class KBMatch:
    path: str
    line: int
    preview: str
    score: float = 0.0


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
        # prune excluded dirs in-place
        dirnames[:] = [d for d in dirnames if d not in exclude]
        for fn in filenames:
            p = Path(dirpath) / fn
            if exts_set and p.suffix.lower() not in exts_set:
                continue
            yield p
            n += 1
            if max_files > 0 and n >= max_files:
                return


def search_codebase(
    query: str,
    *,
    root: str | Path = ".",
    max_matches: int = 30,
    max_files: int = 500,
    max_bytes_per_file: int = 512_000,
    exts: Optional[Sequence[str]] = None,
    exclude_dirs: Optional[Sequence[str]] = None,
    context_lines: int = 1,
    case_sensitive: bool = False,
    group_adjacent: bool = True,
) -> List[KBMatch]:
    """
    Lightweight, dependency-free codebase/KB search.

    This is intentionally simple: it returns line-based matches with previews.
    Use it as the "initial context gatherer" before calling ContextEngine.
    """
    q = (query or "").strip()
    if not q:
        return []

    rootp = Path(root).resolve()
    if not rootp.exists() or not rootp.is_dir():
        raise FileNotFoundError(f"KB root not found or not a directory: {rootp}")

    exts = list(exts) if exts is not None else sorted(DEFAULT_EXTS)
    exclude_dirs = list(exclude_dirs) if exclude_dirs is not None else sorted(DEFAULT_EXCLUDE_DIRS)

    flags = 0 if case_sensitive else re.IGNORECASE
    pat = re.compile(re.escape(q), flags=flags)

    raw: List[KBMatch] = []
    for fp in _iter_files(rootp, exts=exts, exclude_dirs=exclude_dirs, max_files=int(max_files)):
        try:
            if fp.stat().st_size > int(max_bytes_per_file):
                continue
            text = fp.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        # quick binary-ish detection: skip files with lots of NULs
        if "\x00" in text:
            continue

        lines = text.splitlines()
        for i, line in enumerate(lines, start=1):
            if not pat.search(line):
                continue
            start = max(1, i - int(context_lines))
            end = min(len(lines), i + int(context_lines))
            snippet = "\n".join(lines[start - 1 : end]).strip()
            # scoring: prefer exact hits + filename hits
            hit_count = len(pat.findall(snippet))
            fname_bonus = 2.0 if pat.search(str(fp.name)) else 0.0
            score = float(hit_count) + fname_bonus
            raw.append(KBMatch(path=str(fp), line=i, preview=snippet, score=score))
            if int(max_matches) > 0 and len(raw) >= int(max_matches) * 5:
                # we gather a bit extra then rank/dedup
                break

    if not raw:
        return []

    # sort by score desc, then path/line stable
    raw.sort(key=lambda m: (-float(m.score), m.path, int(m.line)))

    # dedup identical previews (common when files have repeated boilerplate)
    seen_prev: set[str] = set()
    deduped: List[KBMatch] = []
    for m in raw:
        key = m.preview.strip()
        if key in seen_prev:
            continue
        seen_prev.add(key)
        deduped.append(m)
        if int(max_matches) > 0 and len(deduped) >= int(max_matches):
            break

    if not group_adjacent:
        return deduped

    # merge adjacent matches within the same file (better context, fewer chunks)
    merged: List[KBMatch] = []
    by_file: Dict[str, List[KBMatch]] = {}
    for m in deduped:
        by_file.setdefault(m.path, []).append(m)

    for path, ms in by_file.items():
        ms.sort(key=lambda m: int(m.line))
        buf: List[KBMatch] = []
        for m in ms:
            if not buf:
                buf = [m]
                continue
            if int(m.line) <= int(buf[-1].line) + (context_lines * 2 + 1):
                buf.append(m)
            else:
                merged.append(_merge_buf(buf))
                buf = [m]
        if buf:
            merged.append(_merge_buf(buf))

    merged.sort(key=lambda m: (-float(m.score), m.path, int(m.line)))
    return merged[: int(max_matches)]


def _merge_buf(buf: List[KBMatch]) -> KBMatch:
    if not buf:
        raise ValueError("empty merge buffer")
    path = buf[0].path
    start_line = min(int(b.line) for b in buf)
    # join previews with separator
    preview = "\n...\n".join(b.preview for b in buf if b.preview).strip()
    score = max(float(b.score) for b in buf)
    return KBMatch(path=path, line=start_line, preview=preview, score=score)


def matches_to_context(matches: Sequence[KBMatch]) -> List[str]:
    """
    Convert matches into context chunks suitable for ContextEngine.
    """
    chunks: List[str] = []
    for m in matches:
        chunks.append(f"Source: {m.path}:{m.line}\n{m.preview}".strip())
    return chunks

