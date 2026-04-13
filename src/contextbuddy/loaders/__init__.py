"""
Unified document loader.

Usage:
    from contextbuddy.loaders import load

    chunks = load("report.pdf")
    chunks = load("https://example.com/page")
    chunks = load("./docs/")
    chunks = load(["a.pdf", "b.txt", "https://example.com"])
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Union

from .pdf import load_pdf
from .web import load_url
from .text import load_text_auto
from .docx import load_docx
from .directory import load_directory

__all__ = ["load", "load_pdf", "load_url", "load_text_auto", "load_docx", "load_directory"]


def load(source: Union[str, Path, Sequence[Union[str, Path]]]) -> List[str]:
    """
    Auto-detect source type and load text chunks.

    Accepts:
    - A file path (PDF, DOCX, TXT, MD, CSV, JSON, etc.)
    - A URL (http:// or https://)
    - A directory path (recursively loads all supported files)
    - A list of any of the above (batch load)
    """
    if isinstance(source, (list, tuple)):
        chunks: List[str] = []
        for item in source:
            chunks.extend(load(item))
        return chunks

    s = str(source)

    if s.startswith("http://") or s.startswith("https://"):
        return load_url(s)

    p = Path(s)

    if p.is_dir():
        return load_directory(p)

    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    ext = p.suffix.lower()
    if ext == ".pdf":
        return load_pdf(p)
    if ext == ".docx":
        return load_docx(p)

    return load_text_auto(p)
