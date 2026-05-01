from __future__ import annotations

from pathlib import Path
from typing import List


def load_pdf(path: str | Path) -> List[str]:
    """
    Extract text from a PDF file and return coherent text chunks.

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
    pages: List[str] = []
    for page in doc:
        text = page.get_text("text").strip()
        if text:
            pages.append(text)
    doc.close()

    # Avoid page-wise chunking (contracts often split clauses across pages).
    # Normalize + re-chunk with SmartChunker so sections/clauses stay together.
    from ..chunking import SmartChunker

    full_text = "\n\n".join(pages).strip()
    if not full_text:
        return []

    return SmartChunker().chunk(full_text, doc_type="pdf")
