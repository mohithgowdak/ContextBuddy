from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional, Sequence


def _split_allowed_roots(s: str) -> list[str]:
    """
    Parse a list of roots from env.

    Windows paths contain ':' in the drive prefix (``C:\\``). Splitting on ':' breaks them,
    so we only use ':' as a separator on non-Windows platforms. Windows uses ';' between paths.
    """
    s = (s or "").strip()
    if not s:
        return []
    if ";" in s:
        raw = [p.strip() for p in s.split(";")]
    elif os.name != "nt" and ":" in s:
        raw = [p.strip() for p in s.split(":")]
    else:
        raw = [s]
    return [p for p in raw if p]


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


def _is_dot_root(root: str | Path) -> bool:
    s = str(root).strip().replace("\\", "/")
    return s in (".", "./", "")


def _resolve_root_path(root: str | Path, *, allowed: list[Path]) -> Path:
    """
    Resolve `root` for MCP tools.

    If the client passes ``.`` and a single allowed root is configured (typical IDE setup),
    use that path instead of the process cwd — stdio MCP hosts do not always set cwd to
    the workspace, so ``Path('.').resolve()`` can point at the user profile (e.g. Windows).
    """
    if allowed and _is_dot_root(root):
        if len(allowed) == 1:
            return allowed[0].resolve()
        if len(allowed) > 1:
            raise PermissionError(
                "root '.' is ambiguous when multiple CONTEXTBUDDY_ALLOWED_ROOTS entries are set; "
                "pass an explicit directory path for root."
            )
    return Path(root).resolve()


def validate_root(root: str | Path, *, allowed_roots: Optional[Sequence[Path]] = None) -> Path:
    """
    Resolve and validate a root directory for MCP.

    - Ensures it exists and is a directory.
    - If an allowlist is provided (or configured via env), enforces containment.
    """
    allow = list(allowed_roots) if allowed_roots is not None else allowed_roots_from_env()
    rp = _resolve_root_path(root, allowed=allow)
    if not rp.exists() or not rp.is_dir():
        raise FileNotFoundError(f"Root not found or not a directory: {rp}")
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

