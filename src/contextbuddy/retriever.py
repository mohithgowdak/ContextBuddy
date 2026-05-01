from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, List, Optional, Sequence, Union

from .engine import ContextEngine, ContextEngineConfig, ContextReport
from .store.memory import MemoryStore, SearchResult
from .types import Embedder, ModelPricing, Tokenizer


def rrf_fuse(
    rankings: Sequence[Sequence[str]],
    *,
    k: int = 60,
    weights: Optional[Sequence[float]] = None,
) -> List[tuple[str, float]]:
    """
    Reciprocal Rank Fusion (RRF) over multiple ranked lists.

    `rankings` is a list of ranked lists (best first). Items are string keys (e.g., file paths).
    Returns a single ranking of (item, fused_score) sorted descending.
    """
    if k <= 0:
        k = 60
    ws = list(weights) if weights is not None else [1.0] * len(rankings)
    if len(ws) != len(rankings):
        ws = [1.0] * len(rankings)

    scores: dict[str, float] = {}
    for w, lst in zip(ws, rankings):
        w = float(w)
        for i, key in enumerate(lst):
            if not key:
                continue
            scores[key] = scores.get(key, 0.0) + (w / (k + i + 1))
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


class Retriever:
    """
    Multi-document RAG in one object.

    Wires together: MemoryStore (search) + ContextEngine (compress) + LLM call.

    Usage:
        store = MemoryStore().add(load("./docs/"))
        retriever = Retriever(store, max_context_tokens=4000, dev_mode=True)
        result = retriever.query("question", llm_call=my_llm)
    """

    def __init__(
        self,
        store: MemoryStore,
        *,
        max_context_tokens: int = 4000,
        min_relevance: float = 0.15,
        top_k: int = 20,
        dev_mode: bool = False,
        pricing: Optional[ModelPricing] = None,
        embedder: Optional[Embedder] = None,
        tokenizer: Optional[Tokenizer] = None,
    ):
        self.store = store
        self.top_k = top_k

        config_kwargs: dict = {
            "max_context_tokens": max_context_tokens,
            "min_relevance": min_relevance,
            "dev_mode": dev_mode,
        }
        if pricing is not None:
            config_kwargs["pricing"] = pricing

        self.engine = ContextEngine(
            ContextEngineConfig(**config_kwargs),
            embedder=embedder,
            tokenizer=tokenizer,
        )
        self.last_report: Optional[ContextReport] = None
        self.last_search_results: List[SearchResult] = []

    def retrieve(self, query: str) -> List[str]:
        """Search the store and return raw matching chunks (pre-compression)."""
        results = self.store.search(query, top_k=self.top_k)
        self.last_search_results = results
        return [r.chunk for r in results]

    async def aretrieve(self, query: str) -> List[str]:
        """Async counterpart of `retrieve`."""
        asearch = getattr(self.store, "asearch", None)
        if asearch is not None:
            results = await asearch(query, top_k=self.top_k)
        else:
            results = await asyncio.to_thread(self.store.search, query, top_k=self.top_k)
        self.last_search_results = results
        return [r.chunk for r in results]

    def build_prompt(self, query: str) -> tuple[str, ContextReport]:
        """Search, compress, and return the final prompt + report."""
        chunks = self.retrieve(query)
        final_prompt, report = self.engine.build_prompt(
            user_prompt=query,
            context=chunks,
        )
        self.last_report = report
        self.engine.last_report = report
        return final_prompt, report

    def query(
        self,
        question: str,
        *,
        llm_call: Callable[[str], Any],
    ) -> Any:
        """Search -> compress -> call LLM -> return response."""
        final_prompt, report = self.build_prompt(question)
        self.last_report = report
        self.engine._emit_report(report)
        return llm_call(final_prompt)

    async def aquery(
        self,
        question: str,
        *,
        llm_call: Callable[[str], Awaitable[Any]],
    ) -> Any:
        """Async version of query(). Retrieval + compression run off the event loop."""
        chunks = await self.aretrieve(question)
        final_prompt, report = await asyncio.to_thread(
            self.engine.build_prompt,
            user_prompt=question,
            context=chunks,
        )
        self.last_report = report
        self.engine.last_report = report
        self.engine._emit_report(report)
        return await llm_call(final_prompt)
