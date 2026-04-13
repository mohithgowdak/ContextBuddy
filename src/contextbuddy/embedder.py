from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import List, Sequence

from .types import Embedder


_word_re = re.compile(r"[A-Za-z0-9_]{2,}")


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
                h = hash(token)
                idx = h % dims
                sign = -1.0 if (h & 1) else 1.0
                vec[idx] += sign
            out.append(_l2_normalize(vec))
        return out


class OpenAIEmbedder:
    """
    Optional embedder using the `openai` Python SDK.

    Install: `pip install contextbuddy[openai]`
    """

    def __init__(self, *, model: str = "text-embedding-3-small", client=None):
        try:
            from openai import OpenAI  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "OpenAIEmbedder requires the 'openai' package. "
                "Install with: pip install contextbuddy[openai]"
            ) from e

        self._model = model
        self._client = client or OpenAI()

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        resp = self._client.embeddings.create(model=self._model, input=list(texts))
        return [d.embedding for d in resp.data]
