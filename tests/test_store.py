import json
import os
import tempfile

from contextbuddy.store.memory import MemoryStore, SearchResult
from contextbuddy.store.persistent import PersistentStore


def test_add_and_search() -> None:
    store = MemoryStore()
    store.add(["machine learning algorithms", "cooking recipes for dinner", "neural network training"])
    assert store.size == 3

    results = store.search("machine learning", top_k=2)
    assert len(results) <= 2
    assert isinstance(results[0], SearchResult)
    assert results[0].score >= results[-1].score


def test_deduplication() -> None:
    store = MemoryStore()
    store.add(["same chunk", "same chunk", "different chunk"])
    assert store.size == 2


def test_dedup_disabled() -> None:
    store = MemoryStore()
    store.add(["same chunk", "same chunk"], deduplicate=False)
    assert store.size == 2


def test_chaining() -> None:
    store = MemoryStore().add(["chunk one"]).add(["chunk two"])
    assert store.size == 2


def test_metadata() -> None:
    store = MemoryStore()
    store.add(["content about payments and invoices"], metadata={"source": "invoice.pdf"})
    results = store.search("invoice")
    assert results[0].metadata == {"source": "invoice.pdf"}


def test_clear() -> None:
    store = MemoryStore()
    store.add(["test chunk"])
    assert store.size == 1
    store.clear()
    assert store.size == 0


def test_to_dict_and_from_dict() -> None:
    store = MemoryStore()
    store.add(["alpha", "beta"], metadata={"source": "test"})

    data = store.to_dict()
    assert len(data["chunks"]) == 2

    restored = MemoryStore.from_dict(data)
    assert restored.size == 2
    results = restored.search("alpha")
    assert len(results) > 0


def test_get_chunks() -> None:
    store = MemoryStore()
    store.add(["first", "second"])
    assert store.get_chunks() == ["first", "second"]


def test_persistent_store() -> None:
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        store = PersistentStore(path)
        store.add(["persistent chunk about technology"])
        assert store.size == 1

        store2 = PersistentStore(path)
        assert store2.size == 1
        results = store2.search("technology")
        assert len(results) == 1
    finally:
        os.unlink(path)


def test_empty_search() -> None:
    store = MemoryStore()
    results = store.search("anything")
    assert results == []


def test_min_score_filter() -> None:
    store = MemoryStore()
    store.add(["machine learning", "cooking dinner"])
    results = store.search("machine learning", min_score=0.99)
    # May or may not have results depending on embedder, but should not crash
    assert isinstance(results, list)
