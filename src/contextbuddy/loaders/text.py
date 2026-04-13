from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import List


def load_text(path: str | Path) -> List[str]:
    """Load a plain text file, splitting on double-newlines."""
    p = Path(path)
    content = p.read_text(encoding="utf-8", errors="replace")
    return _split_paragraphs(content)


def load_markdown(path: str | Path) -> List[str]:
    """Load a Markdown file. Splits on headings and double-newlines."""
    p = Path(path)
    content = p.read_text(encoding="utf-8", errors="replace")
    return _split_paragraphs(content)


def load_csv(path: str | Path) -> List[str]:
    """Load a CSV file. Each row becomes a chunk (header: value pairs)."""
    p = Path(path)
    content = p.read_text(encoding="utf-8", errors="replace")
    reader = csv.reader(io.StringIO(content))
    rows = list(reader)
    if len(rows) < 2:
        return [content.strip()] if content.strip() else []

    header = rows[0]
    chunks: List[str] = []
    for row in rows[1:]:
        pairs = [f"{h}: {v}" for h, v in zip(header, row) if v.strip()]
        if pairs:
            chunks.append(" | ".join(pairs))
    return chunks


def load_json(path: str | Path) -> List[str]:
    """Load a JSON file. Top-level list items become chunks; objects are stringified."""
    p = Path(path)
    content = p.read_text(encoding="utf-8", errors="replace")
    data = json.loads(content)

    if isinstance(data, list):
        return [json.dumps(item, ensure_ascii=False, indent=None) if not isinstance(item, str) else item for item in data]
    if isinstance(data, dict):
        chunks: List[str] = []
        for k, v in data.items():
            if isinstance(v, str):
                chunks.append(f"{k}: {v}")
            else:
                chunks.append(f"{k}: {json.dumps(v, ensure_ascii=False)}")
        return chunks
    return [str(data)]


def load_log(path: str | Path) -> List[str]:
    """Load a log file. Groups consecutive non-blank lines."""
    return load_text(path)


def _split_paragraphs(text: str) -> List[str]:
    import re
    parts = re.split(r"\n\s*\n+", text)
    return [p.strip() for p in parts if p.strip()]


_EXT_MAP = {
    ".txt": load_text,
    ".md": load_markdown,
    ".markdown": load_markdown,
    ".csv": load_csv,
    ".json": load_json,
    ".jsonl": load_text,
    ".log": load_log,
    ".xml": load_text,
    ".yaml": load_text,
    ".yml": load_text,
    ".ini": load_text,
    ".cfg": load_text,
    ".toml": load_text,
    ".rst": load_text,
    ".html": load_text,
    ".htm": load_text,
}


def load_text_auto(path: str | Path) -> List[str]:
    """Auto-detect file type by extension and load accordingly."""
    p = Path(path)
    ext = p.suffix.lower()
    loader = _EXT_MAP.get(ext, load_text)
    return loader(p)
