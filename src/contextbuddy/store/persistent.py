from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from ..types import Embedder
from .memory import MemoryStore


class PersistentStore(MemoryStore):
    """
    MemoryStore backed by a JSON file for persistence across runs.

    Data is loaded on init and saved after every add() call.
    """

    def __init__(self, path: str | Path, *, embedder: Optional[Embedder] = None, auto_save: bool = True):
        super().__init__(embedder=embedder)
        self._path = Path(path)
        self._auto_save = auto_save

        if self._path.exists() and self._path.stat().st_size > 0:
            try:
                raw = json.loads(self._path.read_text(encoding="utf-8"))
                restored = MemoryStore.from_dict(raw, embedder=embedder)
                self._chunks = restored._chunks
                self._vectors = restored._vectors
                self._metadata = restored._metadata
                self._seen_hashes = restored._seen_hashes
            except (json.JSONDecodeError, KeyError):
                pass

    def add(self, chunks, *, metadata=None, deduplicate=True) -> "PersistentStore":
        super().add(chunks, metadata=metadata, deduplicate=deduplicate)
        if self._auto_save:
            self.save()
        return self

    def save(self) -> None:
        """Write current store to disk."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False),
            encoding="utf-8",
        )

    def delete_file(self) -> None:
        """Remove the backing file."""
        if self._path.exists():
            self._path.unlink()
