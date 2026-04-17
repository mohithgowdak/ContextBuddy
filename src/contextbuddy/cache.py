from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


def _hash_key(*parts: str) -> str:
    raw = "|".join(parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    sets: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class EmbeddingCache:
    """
    Cache embedding vectors to avoid re-embedding the same text.

    Keyed on the text content hash. Works in-memory by default;
    optionally persists to a JSON file.
    """

    def __init__(self, *, persist_path: Optional[str | Path] = None):
        self._cache: Dict[str, List[float]] = {}
        self._persist_path = Path(persist_path) if persist_path else None
        self.stats = CacheStats()

        if self._persist_path and self._persist_path.exists():
            try:
                raw = json.loads(self._persist_path.read_text(encoding="utf-8"))
                self._cache = raw.get("embeddings", {})
            except Exception:
                pass

    def get(self, text: str) -> Optional[List[float]]:
        key = _hash_key(text)
        vec = self._cache.get(key)
        if vec is not None:
            self.stats.hits += 1
        else:
            self.stats.misses += 1
        return vec

    def put(self, text: str, vector: List[float]) -> None:
        key = _hash_key(text)
        self._cache[key] = vector
        self.stats.sets += 1

    def get_many(self, texts: Sequence[str]) -> List[Optional[List[float]]]:
        return [self.get(t) for t in texts]

    def put_many(self, texts: Sequence[str], vectors: Sequence[List[float]]) -> None:
        for t, v in zip(texts, vectors):
            self.put(t, v)

    def save(self) -> None:
        if self._persist_path:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            self._persist_path.write_text(
                json.dumps({"embeddings": self._cache}, ensure_ascii=False),
                encoding="utf-8",
            )

    @property
    def size(self) -> int:
        return len(self._cache)

    def clear(self) -> None:
        self._cache.clear()
        self.stats = CacheStats()


class ResponseCache:
    """
    Cache LLM responses keyed on (compressed_prompt, model).

    Useful in dev/testing to avoid repeated LLM calls for the same input.
    Supports TTL (time-to-live) in seconds.
    """

    def __init__(self, *, ttl_seconds: float = 3600.0, persist_path: Optional[str | Path] = None):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._ttl = ttl_seconds
        self._persist_path = Path(persist_path) if persist_path else None
        self.stats = CacheStats()

        if self._persist_path and self._persist_path.exists():
            try:
                raw = json.loads(self._persist_path.read_text(encoding="utf-8"))
                self._cache = raw.get("responses", {})
            except Exception:
                pass

    def get(self, prompt: str, model: str = "") -> Optional[Any]:
        key = _hash_key(prompt, model)
        entry = self._cache.get(key)
        if entry is None:
            self.stats.misses += 1
            return None
        if time.time() - entry.get("ts", 0) > self._ttl:
            del self._cache[key]
            self.stats.misses += 1
            return None
        self.stats.hits += 1
        return entry.get("value")

    def put(self, prompt: str, model: str, value: Any) -> None:
        key = _hash_key(prompt, model)
        self._cache[key] = {"value": value, "ts": time.time()}
        self.stats.sets += 1

    def save(self) -> None:
        if self._persist_path:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            self._persist_path.write_text(
                json.dumps({"responses": self._cache}, ensure_ascii=False, default=str),
                encoding="utf-8",
            )

    @property
    def size(self) -> int:
        return len(self._cache)

    def clear(self) -> None:
        self._cache.clear()
        self.stats = CacheStats()


class CachedEmbedder:
    """
    Wraps any Embedder with an EmbeddingCache. Drop-in replacement.
    """

    def __init__(self, embedder: Any, cache: EmbeddingCache):
        self._embedder = embedder
        self._cache = cache

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        results: List[Optional[List[float]]] = self._cache.get_many(texts)
        miss_indices: List[int] = []
        miss_texts: List[str] = []

        for i, r in enumerate(results):
            if r is None:
                miss_indices.append(i)
                miss_texts.append(texts[i])

        if miss_texts:
            new_vecs = self._embedder.embed(miss_texts)
            self._cache.put_many(miss_texts, new_vecs)
            for i, vec in zip(miss_indices, new_vecs):
                results[i] = vec

        return results  # type: ignore[return-value]

    async def aembed(self, texts: Sequence[str]) -> List[List[float]]:
        results: List[Optional[List[float]]] = self._cache.get_many(texts)
        miss_indices: List[int] = []
        miss_texts: List[str] = []

        for i, r in enumerate(results):
            if r is None:
                miss_indices.append(i)
                miss_texts.append(texts[i])

        if miss_texts:
            aembed_fn = getattr(self._embedder, "aembed", None)
            if aembed_fn is not None:
                new_vecs = await aembed_fn(miss_texts)
            else:
                new_vecs = await asyncio.to_thread(self._embedder.embed, miss_texts)
            self._cache.put_many(miss_texts, new_vecs)
            for i, vec in zip(miss_indices, new_vecs):
                results[i] = vec

        return results  # type: ignore[return-value]

    @property
    def stats(self) -> CacheStats:
        return self._cache.stats
