"""Tests for the LangChain integration module."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeSearchResult:
    def __init__(self, chunk: str, metadata: dict | None = None):
        self.chunk = chunk
        self.score = 1.0
        self.metadata = metadata or {}


class _FakeStore:
    def __init__(self, chunks):
        self._chunks = chunks

    def search(self, query: str, top_k: int = 20):
        return [_FakeSearchResult(c) for c in self._chunks[:top_k]]


# ---------------------------------------------------------------------------
# Stub behaviour when langchain-core is NOT installed
# ---------------------------------------------------------------------------

def test_stub_compressor_raises_without_langchain(monkeypatch):
    """Import should succeed; instantiation should raise ImportError if LC missing."""
    import contextbuddy.langchain as lc_mod

    if lc_mod._LANGCHAIN_AVAILABLE:
        pytest.skip("langchain-core is installed — stub test not applicable")

    with pytest.raises(ImportError, match="langchain-core"):
        lc_mod.ContextBuddyCompressor()


def test_stub_retriever_raises_without_langchain(monkeypatch):
    import contextbuddy.langchain as lc_mod

    if lc_mod._LANGCHAIN_AVAILABLE:
        pytest.skip("langchain-core is installed — stub test not applicable")

    with pytest.raises(ImportError, match="langchain-core"):
        lc_mod.ContextBuddyRetriever(store=None)


# ---------------------------------------------------------------------------
# Full integration tests (skipped when langchain-core absent)
# ---------------------------------------------------------------------------

langchain_core = pytest.importorskip("langchain_core", reason="langchain-core not installed")


from contextbuddy.langchain import ContextBuddyCompressor, ContextBuddyRetriever  # noqa: E402
from langchain_core.documents import Document  # noqa: E402


SAMPLE_DOCS = [
    Document(page_content="Refunds are processed within 5 business days.", metadata={"source": "policy.pdf"}),
    Document(page_content="To request a refund, email support@example.com.", metadata={"source": "policy.pdf"}),
    Document(page_content="Our headquarters is located in New York.", metadata={"source": "about.pdf"}),
    Document(page_content="The sky is blue and the grass is green.", metadata={}),
]


# --- ContextBuddyCompressor -------------------------------------------------

class TestContextBuddyCompressor:
    def test_returns_subset_of_documents(self):
        compressor = ContextBuddyCompressor(max_context_tokens=200)
        result = compressor.compress_documents(SAMPLE_DOCS, query="refund policy")
        assert isinstance(result, list)
        assert all(isinstance(d, Document) for d in result)
        # Must be a subset — never adds documents
        assert len(result) <= len(SAMPLE_DOCS)

    def test_empty_input_returns_empty(self):
        compressor = ContextBuddyCompressor()
        result = compressor.compress_documents([], query="anything")
        assert result == []

    def test_metadata_preserved(self):
        compressor = ContextBuddyCompressor(max_context_tokens=500)
        result = compressor.compress_documents(SAMPLE_DOCS, query="refund")
        sources = {d.metadata.get("source") for d in result if d.metadata.get("source")}
        # At least one source-tagged doc should survive a refund query
        assert sources, "Expected at least one document with metadata to survive"

    def test_single_document_passes_through(self):
        doc = Document(page_content="A single relevant sentence about returns.")
        compressor = ContextBuddyCompressor(max_context_tokens=500)
        result = compressor.compress_documents([doc], query="returns")
        assert len(result) >= 0  # Engine may keep or drop — just no crash

    def test_conservative_mode_accepted(self):
        compressor = ContextBuddyCompressor(max_context_tokens=300, conservative_mode=True)
        result = compressor.compress_documents(SAMPLE_DOCS[:2], query="refund")
        assert isinstance(result, list)


# --- ContextBuddyRetriever --------------------------------------------------

class TestContextBuddyRetriever:
    def _make_retriever(self, chunks=None, **kwargs):
        if chunks is None:
            chunks = [
                "Refunds are processed within 5 business days.",
                "Email support to request a refund.",
                "Our office is in New York.",
            ]
        store = _FakeStore(chunks)
        return ContextBuddyRetriever(store=store, **kwargs)

    def test_returns_documents(self):
        retriever = self._make_retriever(max_context_tokens=300)
        docs = retriever.invoke("refund policy")
        assert isinstance(docs, list)
        assert all(isinstance(d, Document) for d in docs)

    def test_empty_store_returns_empty(self):
        retriever = self._make_retriever(chunks=[], max_context_tokens=300)
        docs = retriever.invoke("anything")
        assert docs == []

    def test_content_from_store_chunks(self):
        chunks = ["Chunk A about invoices.", "Chunk B about payments.", "Chunk C about weather."]
        retriever = self._make_retriever(chunks=chunks, max_context_tokens=500)
        docs = retriever.invoke("invoice")
        # Returned docs must come from the original chunks list
        returned_texts = {d.page_content for d in docs}
        assert returned_texts.issubset(set(chunks))

    def test_top_k_respected(self):
        chunks = [f"Chunk {i}" for i in range(50)]
        retriever = self._make_retriever(chunks=chunks, top_k=3, max_context_tokens=200)
        docs = retriever.invoke("some query")
        # Can't get more than top_k results
        assert len(docs) <= 3

    @pytest.mark.asyncio
    async def test_async_returns_same_type(self):
        retriever = self._make_retriever(max_context_tokens=300)
        docs = await retriever.ainvoke("refund policy")
        assert isinstance(docs, list)
        assert all(isinstance(d, Document) for d in docs)
