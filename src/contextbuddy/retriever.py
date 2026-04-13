from __future__ import annotations

from typing import Any, Awaitable, Callable, List, Optional, Sequence, Union

from .engine import ContextEngine, ContextEngineConfig, ContextReport
from .store.memory import MemoryStore, SearchResult
from .types import Embedder, ModelPricing, Tokenizer


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
        """Async version of query()."""
        final_prompt, report = self.build_prompt(question)
        self.last_report = report
        self.engine._emit_report(report)
        return await llm_call(final_prompt)
