from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

_DEFAULT_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".csv", ".json", ".jsonl",
    ".log", ".xml", ".yaml", ".yml", ".rst", ".html", ".htm",
    ".pdf", ".docx",
    # Codebases (python-only for now)
    ".py",
}


def load_directory(
    path: str | Path,
    *,
    extensions: Optional[Sequence[str]] = None,
    max_depth: int = 5,
    max_file_bytes: int = 10 * 1024 * 1024,
    prefix_source: bool = True,
) -> List[str]:
    """
    Recursively load all supported files from a directory.

    Each chunk is prefixed with the source filename for traceability.
    """
    root = Path(path)
    if not root.is_dir():
        raise NotADirectoryError(f"Not a directory: {root}")

    allowed = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in (extensions or _DEFAULT_EXTENSIONS)}
    all_chunks: List[str] = []

    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix.lower() not in allowed:
            continue

        rel = p.relative_to(root)
        depth = len(rel.parts) - 1
        if depth > max_depth:
            continue

        try:
            if p.stat().st_size > max_file_bytes:
                continue
        except OSError:
            continue

        try:
            chunks = _load_single(p)
        except Exception:
            continue

        if prefix_source:
            prefix = f"[{rel}] "
            chunks = [prefix + c for c in chunks]

        all_chunks.extend(chunks)

    return all_chunks


def _load_single(p: Path) -> List[str]:
    ext = p.suffix.lower()
    if ext == ".pdf":
        from .pdf import load_pdf
        return load_pdf(p)
    if ext == ".docx":
        from .docx import load_docx
        return load_docx(p)
    from .text import load_text_auto
    return load_text_auto(p)
