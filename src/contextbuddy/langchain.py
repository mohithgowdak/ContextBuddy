"""LangChain integration: ContextBuddyCompressor and ContextBuddyRetriever."""
from __future__ import annotations

from typing import Any, List, Optional, Sequence

try:
    from langchain_core.documents import Document
    from langchain_core.documents.compressor import BaseDocumentCompressor
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.callbacks.manager import (
        CallbackManagerForRetrieverRun,
        AsyncCallbackManagerForRetrieverRun,
    )
    _LANGCHAIN_AVAILABLE = True
except ImportError:
    _LANGCHAIN_AVAILABLE = False

from .engine import ContextEngine, ContextEngineConfig


def _require_langchain() -> None:
    if not _LANGCHAIN_AVAILABLE:
        raise ImportError(
            "langchain-core is required for LangChain integration. "
            'Install it with: pip install "contextbuddy[langchain]"'
        )


if _LANGCHAIN_AVAILABLE:

    class ContextBuddyCompressor(BaseDocumentCompressor):
        """LangChain document compressor that uses ContextBuddy to trim a list of
        retrieved documents to fit a token budget.

        Use this as the ``base_compressor`` in a ``ContextualCompressionRetriever``
        to automatically prune over-retrieved chunks before they reach the LLM.

        Example::

            from contextbuddy import ContextBuddyCompressor
            from langchain_core.retrievers import YourBaseRetriever
            from langchain.retrievers import ContextualCompressionRetriever

            compressor = ContextBuddyCompressor(max_context_tokens=3000)
            retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=YourBaseRetriever(),
            )
        """

        max_context_tokens: int = 2000
        min_relevance: float = 0.15
        conservative_mode: bool = False

        class Config:
            arbitrary_types_allowed = True

        def compress_documents(
            self,
            documents: Sequence[Document],
            query: str,
            callbacks: Any = None,
        ) -> List[Document]:
            if not documents:
                return []

            cfg = ContextEngineConfig(
                max_context_tokens=self.max_context_tokens,
                min_relevance=self.min_relevance,
                conservative_mode=self.conservative_mode,
                dev_mode=False,
                rich_output=False,
            )
            engine = ContextEngine(cfg)

            chunks = [doc.page_content for doc in documents]
            _, report = engine.build_prompt(user_prompt=query, context=chunks)

            return [documents[i] for i in report.selected_indices]

    class ContextBuddyRetriever(BaseRetriever):
        """LangChain retriever backed by a ContextBuddy MemoryStore + compression.

        Loads an existing ``MemoryStore`` (or any object with a ``.search(query,
        top_k)`` method returning objects with a ``.chunk`` attribute), runs a
        semantic search, then compresses the hits with ContextBuddy before
        returning them as ``Document`` objects.

        Example::

            from contextbuddy import MemoryStore, ContextBuddyRetriever

            store = MemoryStore()
            store.add(["chunk 1 ...", "chunk 2 ..."])

            retriever = ContextBuddyRetriever(store=store, max_context_tokens=2000)
            docs = retriever.invoke("what is the refund policy?")
        """

        store: Any  # MemoryStore or any compatible search store
        max_context_tokens: int = 2000
        min_relevance: float = 0.15
        top_k: int = 20

        class Config:
            arbitrary_types_allowed = True

        def _get_relevant_documents(
            self,
            query: str,
            *,
            run_manager: CallbackManagerForRetrieverRun,
        ) -> List[Document]:
            results = self.store.search(query, top_k=self.top_k)
            if not results:
                return []

            chunks = [r.chunk for r in results]
            metadata_list = [getattr(r, "metadata", {}) or {} for r in results]

            cfg = ContextEngineConfig(
                max_context_tokens=self.max_context_tokens,
                min_relevance=self.min_relevance,
                dev_mode=False,
                rich_output=False,
            )
            engine = ContextEngine(cfg)
            _, report = engine.build_prompt(user_prompt=query, context=chunks)

            return [
                Document(page_content=chunks[i], metadata=metadata_list[i])
                for i in report.selected_indices
            ]

        async def _aget_relevant_documents(
            self,
            query: str,
            *,
            run_manager: AsyncCallbackManagerForRetrieverRun,
        ) -> List[Document]:
            # Compression is synchronous; satisfy the async contract without blocking
            # the event loop by delegating to a thread.
            import asyncio
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                lambda: self._get_relevant_documents(
                    query, run_manager=run_manager  # type: ignore[arg-type]
                ),
            )

else:
    # Provide stub classes so that `from contextbuddy import ContextBuddyCompressor`
    # at the module level does not crash — the error is deferred until instantiation.

    class ContextBuddyCompressor:  # type: ignore[no-redef]
        """Stub: install langchain-core to use this class."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _require_langchain()

    class ContextBuddyRetriever:  # type: ignore[no-redef]
        """Stub: install langchain-core to use this class."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _require_langchain()


__all__ = ["ContextBuddyCompressor", "ContextBuddyRetriever"]
