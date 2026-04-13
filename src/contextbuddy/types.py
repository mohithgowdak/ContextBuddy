from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Protocol, Sequence, Tuple


class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> List[List[float]]: ...


class Tokenizer(Protocol):
    def count_tokens(self, text: str) -> int: ...


@dataclass(frozen=True)
class ModelPricing:
    """
    Simple per-1k token pricing model.
    """

    input_per_1k: float
    output_per_1k: float = 0.0


@dataclass(frozen=True)
class CostEstimate:
    input_cost: float
    output_cost: float

    @property
    def total_cost(self) -> float:
        return self.input_cost + self.output_cost


CostModel = Callable[[int, int], CostEstimate]
