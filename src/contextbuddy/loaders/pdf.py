from __future__ import annotations

from pathlib import Path
from typing import List


def load_pdf(path: str | Path) -> List[str]:
    """
    Extract text from a PDF file, one chunk per page.

    Requires: pip install "contextbuddy[pdf]"
    """
    try:
        import pymupdf  # type: ignore
    except ImportError as e:
        raise ImportError(
            "PDF loading requires 'pymupdf'. "
            "Install with: pip install \"contextbuddy[pdf]\""
        ) from e

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"PDF not found: {p}")

    doc = pymupdf.open(str(p))
    chunks: List[str] = []
    for page in doc:
        text = page.get_text("text").strip()
        if text:
            chunks.append(text)
    doc.close()
    return chunks
