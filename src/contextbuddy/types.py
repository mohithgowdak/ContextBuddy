from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Protocol, Sequence, Tuple


class Embedder(Protocol):
    """Public embedder contract. Duck-typed — no inheritance required.

    Any object with `embed(texts: Sequence[str]) -> List[List[float]]`
    works. Raising `ImportError` in the adapter's `__init__` is the
    canonical way to gate optional dependencies; see `OpenAIEmbedder`.
    """

    def embed(self, texts: Sequence[str]) -> List[List[float]]: ...


class AsyncEmbedder(Protocol):
    """Optional async embedder contract.

    Concrete adapters in this repo provide `aembed` by wrapping `embed`
    with `asyncio.to_thread`. Network-backed embedders (e.g. OpenAI) may
    override `aembed` with a native async implementation.
    """

    async def aembed(self, texts: Sequence[str]) -> List[List[float]]: ...


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
