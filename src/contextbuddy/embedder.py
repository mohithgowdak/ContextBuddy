from __future__ import annotations

import asyncio
import hashlib
import math
import re
from dataclasses import dataclass
from typing import List, Sequence

from .types import Embedder


_word_re = re.compile(r"[A-Za-z0-9_]{2,}")


def _token_hash(token: str) -> int:
    """
    Deterministic, process-independent token hash.

    Python's built-in `hash()` is randomized per interpreter process
    (via PYTHONHASHSEED), which would make `LocalHashEmbedder` produce
    different vectors across runs — violating the determinism red line.
    `hashlib.md5` is stable everywhere.
    """
    return int.from_bytes(hashlib.md5(token.encode("utf-8")).digest()[:8], "big")


def _l2_normalize(vec: List[float]) -> List[float]:
    n2 = sum(v * v for v in vec)
    if n2 <= 0.0:
        return vec
    inv = 1.0 / math.sqrt(n2)
    return [v * inv for v in vec]


@dataclass(frozen=True)
class LocalHashEmbedder:
    """
    Offline, dependency-free embedder using a hashing trick over word tokens.

    It is not a true semantic embedding model, but it provides a surprisingly
    useful relevance signal for paragraph pruning and keeps the library
    zero-friction out of the box.

    Deterministic across processes (uses `hashlib.md5`, not `hash()`).
    """

    dims: int = 256

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        out: List[List[float]] = []
        dims = int(self.dims)
        if dims <= 0:
            raise ValueError("dims must be > 0")

        for text in texts:
            vec = [0.0] * dims
            for m in _word_re.finditer(text.lower()):
                token = m.group(0)
                h = _token_hash(token)
                idx = h % dims
                sign = -1.0 if (h & 1) else 1.0
                vec[idx] += sign
            out.append(_l2_normalize(vec))
        return out

    async def aembed(self, texts: Sequence[str]) -> List[List[float]]:
        return await asyncio.to_thread(self.embed, texts)


class OpenAIEmbedder:
    """
    Optional embedder using the `openai` Python SDK.

    Install: `pip install "contextbuddy[openai]"`

    Pass any OpenAI-compatible client via `client=` to reuse your own
    retry/timeout configuration. Pass an `AsyncOpenAI` instance to get
    a true-async `aembed`; otherwise `aembed` falls back to running the
    sync path in a worker thread.
    """

    def __init__(self, *, model: str = "text-embedding-3-small", client=None):
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as e:
            raise ImportError(
                "OpenAIEmbedder requires the 'openai' package. "
                "Install with: pip install \"contextbuddy[openai]\""
            ) from e

        self._model = model
        self._client = client or OpenAI()

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        resp = self._client.embeddings.create(model=self._model, input=list(texts))
        return [d.embedding for d in resp.data]

    async def aembed(self, texts: Sequence[str]) -> List[List[float]]:
        create = getattr(getattr(self._client, "embeddings", None), "create", None)
        if create is not None and asyncio.iscoroutinefunction(create):
            resp = await create(model=self._model, input=list(texts))
            return [d.embedding for d in resp.data]
        return await asyncio.to_thread(self.embed, texts)
