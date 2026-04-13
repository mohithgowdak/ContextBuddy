"""
Pipeline: the one-liner that wires loaders + store + retriever + compressor + router.

Usage:
    pipeline = Pipeline.from_directory("./docs/", dev_mode=True)
    result = pipeline.query("Summarize the contract", llm_call=my_llm)

    # Or with auto-routing:
    result = pipeline.query("Summarize the contract", llm_calls={"gpt-4o": expensive_fn, "gpt-4o-mini": cheap_fn})
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Union

from .cache import CachedEmbedder, EmbeddingCache, ResponseCache
from .engine import ContextEngine, ContextEngineConfig, ContextReport
from .loaders import load
from .retriever import Retriever
from .router import Router, RouteRule
from .store.memory import MemoryStore
from .store.persistent import PersistentStore
from .types import Embedder, ModelPricing, Tokenizer


class Pipeline:
    """
    All-in-one context pipeline.

    Combines: loaders -> store -> retriever -> compressor -> router -> cache.
    Every component is optional and configurable.
    """

    def __init__(
        self,
        *,
        store: Optional[MemoryStore] = None,
        max_context_tokens: int = 4000,
        min_relevance: float = 0.15,
        top_k: int = 20,
        dev_mode: bool = False,
        pricing: Optional[ModelPricing] = None,
        embedder: Optional[Embedder] = None,
        tokenizer: Optional[Tokenizer] = None,
        router: Optional[Router] = None,
        embedding_cache: Optional[EmbeddingCache] = None,
        response_cache: Optional[ResponseCache] = None,
    ):
        actual_embedder = embedder
        if embedding_cache and actual_embedder:
            actual_embedder = CachedEmbedder(actual_embedder, embedding_cache)
        elif embedding_cache:
            from .embedder import LocalHashEmbedder
            actual_embedder = CachedEmbedder(LocalHashEmbedder(), embedding_cache)

        self.store = store or MemoryStore(embedder=actual_embedder)
        self.router = router
        self.response_cache = response_cache
        self._embedding_cache = embedding_cache

        config_kwargs: dict = {
            "max_context_tokens": max_context_tokens,
            "min_relevance": min_relevance,
            "dev_mode": dev_mode,
        }
        if pricing is not None:
            config_kwargs["pricing"] = pricing

        self.retriever = Retriever(
            self.store,
            max_context_tokens=max_context_tokens,
            min_relevance=min_relevance,
            top_k=top_k,
            dev_mode=dev_mode,
            pricing=pricing,
            embedder=actual_embedder,
            tokenizer=tokenizer,
        )

    @property
    def last_report(self) -> Optional[ContextReport]:
        return self.retriever.last_report

    def add(
        self,
        source: Union[str, Path, Sequence[Union[str, Path]]],
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "Pipeline":
        """Load source(s) and add to the store. Returns self for chaining."""
        s = str(source) if isinstance(source, Path) else source

        if isinstance(s, str) and not s.startswith("http") and Path(s).is_dir():
            chunks = load(s)
            self.store.add(chunks, metadata=metadata or {"source": s})
        elif isinstance(s, str) and not s.startswith("http") and Path(s).is_file():
            chunks = load(s)
            self.store.add(chunks, metadata=metadata or {"source": s})
        elif isinstance(s, str) and s.startswith("http"):
            chunks = load(s)
            self.store.add(chunks, metadata=metadata or {"source": s})
        elif isinstance(s, (list, tuple)):
            for item in s:
                self.add(item, metadata=metadata)
        else:
            chunks = load(s)
            self.store.add(chunks, metadata=metadata or {"source": str(s)})

        return self

    def query(
        self,
        question: str,
        *,
        llm_call: Optional[Callable[[str], Any]] = None,
        llm_calls: Optional[Dict[str, Callable[[str], Any]]] = None,
    ) -> Any:
        """
        Search -> compress -> (optionally route) -> call LLM.

        Args:
            llm_call: Single LLM function. Used directly.
            llm_calls: Dict of {model_name: callable}. Router picks the best model.
        """
        if self.response_cache:
            cached = self.response_cache.get(question)
            if cached is not None:
                return cached

        actual_call = llm_call
        if llm_calls and self.router:
            model_name, pricing = self.router.select(question)
            actual_call = llm_calls.get(model_name)
            if actual_call is None:
                actual_call = next(iter(llm_calls.values()))

        if actual_call is None:
            raise ValueError("Provide llm_call or llm_calls with a router.")

        result = self.retriever.query(question, llm_call=actual_call)

        if self.response_cache:
            self.response_cache.put(question, "", result)

        return result

    async def aquery(
        self,
        question: str,
        *,
        llm_call: Optional[Callable[[str], Awaitable[Any]]] = None,
        llm_calls: Optional[Dict[str, Callable[[str], Awaitable[Any]]]] = None,
    ) -> Any:
        """Async version of query()."""
        if self.response_cache:
            cached = self.response_cache.get(question)
            if cached is not None:
                return cached

        actual_call = llm_call
        if llm_calls and self.router:
            model_name, _ = self.router.select(question)
            actual_call = llm_calls.get(model_name)
            if actual_call is None:
                actual_call = next(iter(llm_calls.values()))

        if actual_call is None:
            raise ValueError("Provide llm_call or llm_calls with a router.")

        result = await self.retriever.aquery(question, llm_call=actual_call)

        if self.response_cache:
            self.response_cache.put(question, "", result)

        return result

    @classmethod
    def from_directory(
        cls,
        path: str | Path,
        *,
        dev_mode: bool = False,
        max_context_tokens: int = 4000,
        persist_store: Optional[str | Path] = None,
        **kwargs: Any,
    ) -> "Pipeline":
        """
        One-liner: load a directory into a pipeline, ready to query.
        """
        store: MemoryStore
        if persist_store:
            store = PersistentStore(persist_store)
        else:
            store = MemoryStore()

        pipeline = cls(
            store=store,
            dev_mode=dev_mode,
            max_context_tokens=max_context_tokens,
            **kwargs,
        )
        pipeline.add(path)
        return pipeline

    @classmethod
    def from_sources(
        cls,
        sources: Sequence[Union[str, Path]],
        *,
        dev_mode: bool = False,
        max_context_tokens: int = 4000,
        **kwargs: Any,
    ) -> "Pipeline":
        """
        One-liner: load multiple sources (files, URLs, dirs) into a pipeline.
        """
        pipeline = cls(
            dev_mode=dev_mode,
            max_context_tokens=max_context_tokens,
            **kwargs,
        )
        for s in sources:
            pipeline.add(s)
        return pipeline
