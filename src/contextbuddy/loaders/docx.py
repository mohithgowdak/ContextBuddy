from __future__ import annotations

from pathlib import Path
from typing import List

from ..chunking import SmartChunker


def load_docx(path: str | Path) -> List[str]:
    """
    Extract text from a DOCX file, one chunk per paragraph.

    Requires: pip install "contextbuddy[docx]"
    """
    try:
        from docx import Document  # type: ignore
    except ImportError as e:
        raise ImportError(
            "DOCX loading requires 'python-docx'. "
            "Install with: pip install \"contextbuddy[docx]\""
        ) from e

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"DOCX not found: {p}")

    doc = Document(str(p))
    lines: List[str] = []
    for para in doc.paragraphs:
        t = para.text.strip()
        if t:
            lines.append(t)
        else:
            lines.append("")

    text = "\n".join(lines).strip()
    if not text:
        return []

    return SmartChunker().chunk(text, doc_type="auto")
