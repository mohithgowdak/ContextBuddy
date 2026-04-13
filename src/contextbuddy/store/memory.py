from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..embedder import LocalHashEmbedder
from ..types import Embedder


@dataclass
class SearchResult:
    chunk: str
    score: float
    index: int
    metadata: Dict[str, Any]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / math.sqrt(na * nb)


def _chunk_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


class MemoryStore:
    """
    Zero-dependency in-memory vector store.

    Uses the same embedder as ContextEngine. Pure-Python cosine search
    -- fast enough for <100k chunks, which covers most use cases.
    """

    def __init__(self, *, embedder: Optional[Embedder] = None):
        self._embedder = embedder or LocalHashEmbedder()
        self._chunks: List[str] = []
        self._vectors: List[List[float]] = []
        self._metadata: List[Dict[str, Any]] = []
        self._seen_hashes: set[str] = set()

    @property
    def size(self) -> int:
        return len(self._chunks)

    def add(
        self,
        chunks: str | Sequence[str],
        *,
        metadata: Optional[Dict[str, Any]] = None,
        deduplicate: bool = True,
    ) -> "MemoryStore":
        """
        Add chunks to the store. Returns self for chaining.
        """
        if isinstance(chunks, str):
            chunks = [chunks]

        new_chunks: List[str] = []
        new_meta: List[Dict[str, Any]] = []

        for c in chunks:
            c = str(c).strip()
            if not c:
                continue
            if deduplicate:
                h = _chunk_hash(c)
                if h in self._seen_hashes:
                    continue
                self._seen_hashes.add(h)
            new_chunks.append(c)
            new_meta.append(metadata or {})

        if new_chunks:
            vecs = self._embedder.embed(new_chunks)
            self._chunks.extend(new_chunks)
            self._vectors.extend(vecs)
            self._metadata.extend(new_meta)

        return self

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        min_score: float = 0.0,
    ) -> List[SearchResult]:
        """
        Semantic search. Returns top-k results ranked by cosine similarity.
        """
        if not self._chunks:
            return []

        q_vec = self._embedder.embed([query])[0]
        scored: List[Tuple[float, int]] = []
        for i, v in enumerate(self._vectors):
            s = _cosine(q_vec, v)
            if s >= min_score:
                scored.append((s, i))

        scored.sort(key=lambda x: x[0], reverse=True)

        results: List[SearchResult] = []
        for score, idx in scored[:top_k]:
            results.append(SearchResult(
                chunk=self._chunks[idx],
                score=score,
                index=idx,
                metadata=self._metadata[idx],
            ))
        return results

    def get_chunks(self, *, as_list: bool = True) -> List[str]:
        """Return all stored chunks."""
        return list(self._chunks)

    def clear(self) -> None:
        self._chunks.clear()
        self._vectors.clear()
        self._metadata.clear()
        self._seen_hashes.clear()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize store contents for persistence."""
        return {
            "chunks": self._chunks,
            "vectors": self._vectors,
            "metadata": self._metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], *, embedder: Optional[Embedder] = None) -> "MemoryStore":
        """Restore a store from serialized data (skips re-embedding)."""
        store = cls(embedder=embedder)
        store._chunks = data.get("chunks", [])
        store._vectors = data.get("vectors", [])
        store._metadata = data.get("metadata", [])
        store._seen_hashes = {_chunk_hash(c) for c in store._chunks}
        return store
