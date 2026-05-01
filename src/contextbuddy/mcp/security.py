from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional, Sequence


def _split_allowed_roots(s: str) -> list[str]:
    # Windows commonly uses ';', posix uses ':'. Accept both.
    raw = []
    for part in s.replace(":", ";").split(";"):
        part = (part or "").strip()
        if part:
            raw.append(part)
    return raw


def allowed_roots_from_env(env_var: str = "CONTEXTBUDDY_ALLOWED_ROOTS") -> list[Path]:
    """
    Returns an allowlist of roots if configured by env var.
    If not set, returns [] (meaning: do not enforce allowlist).
    """
    val = (os.environ.get(env_var) or "").strip()
    if not val:
        return []
    roots: list[Path] = []
    for r in _split_allowed_roots(val):
        try:
            roots.append(Path(r).resolve())
        except Exception:
            continue
    return roots


def validate_root(root: str | Path, *, allowed_roots: Optional[Sequence[Path]] = None) -> Path:
    """
    Resolve and validate a root directory for MCP.

    - Ensures it exists and is a directory.
    - If an allowlist is provided (or configured via env), enforces containment.
    """
    rp = Path(root).resolve()
    if not rp.exists() or not rp.is_dir():
        raise FileNotFoundError(f"Root not found or not a directory: {rp}")

    allow = list(allowed_roots) if allowed_roots is not None else allowed_roots_from_env()
    if allow:
        ok = any(_is_under(rp, a) for a in allow)
        if not ok:
            joined = "; ".join(str(a) for a in allow)
            raise PermissionError(f"Root is not in allowed roots. root={rp} allowed={joined}")
    return rp


def resolve_under_root(path: str | Path, *, root: Path) -> Path:
    """
    Resolve a file path and ensure it stays under `root` (path traversal guard).
    """
    p = Path(path)
    if not p.is_absolute():
        p = root / p
    rp = p.resolve()
    if not _is_under(rp, root):
        raise PermissionError(f"Path escapes root: {rp} (root={root})")
    return rp


def _is_under(p: Path, root: Path) -> bool:
    try:
        p.relative_to(root)
        return True
    except Exception:
        return False

