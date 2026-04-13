from __future__ import annotations

from pathlib import Path
from typing import List


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
    chunks: List[str] = []
    buf: List[str] = []

    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            if buf:
                chunks.append("\n".join(buf))
                buf = []
        else:
            buf.append(text)

    if buf:
        chunks.append("\n".join(buf))

    return [c for c in chunks if len(c) >= 20]
