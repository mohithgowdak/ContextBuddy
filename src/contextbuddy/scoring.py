from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from .types import Embedder


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError("Vectors must have same dimensionality")
    dot = 0.0
    na2 = 0.0
    nb2 = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na2 += x * x
        nb2 += y * y
    if na2 <= 0.0 or nb2 <= 0.0:
        return 0.0
    return dot / math.sqrt(na2 * nb2)


@dataclass(frozen=True)
class SemanticScorer:
    embedder: Embedder

    def score(self, *, query: str, chunks: Sequence[str]) -> List[float]:
        if not chunks:
            return []
        vecs = self.embedder.embed([query, *chunks])
        q = vecs[0]
        return [cosine_similarity(q, v) for v in vecs[1:]]

