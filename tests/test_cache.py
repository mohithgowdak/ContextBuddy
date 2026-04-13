import os
import tempfile
import time

from contextbuddy.cache import EmbeddingCache, ResponseCache, CachedEmbedder
from contextbuddy.embedder import LocalHashEmbedder


def test_embedding_cache_hit_miss() -> None:
    cache = EmbeddingCache()
    assert cache.get("hello") is None
    assert cache.stats.misses == 1

    cache.put("hello", [1.0, 2.0, 3.0])
    result = cache.get("hello")
    assert result == [1.0, 2.0, 3.0]
    assert cache.stats.hits == 1


def test_embedding_cache_many() -> None:
    cache = EmbeddingCache()
    cache.put_many(["a", "b"], [[1.0], [2.0]])
    results = cache.get_many(["a", "b", "c"])
    assert results[0] == [1.0]
    assert results[1] == [2.0]
    assert results[2] is None


def test_embedding_cache_persist() -> None:
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        cache = EmbeddingCache(persist_path=path)
        cache.put("test", [1.0, 2.0])
        cache.save()

        cache2 = EmbeddingCache(persist_path=path)
        assert cache2.get("test") == [1.0, 2.0]
    finally:
        os.unlink(path)


def test_response_cache_hit_miss() -> None:
    cache = ResponseCache(ttl_seconds=10.0)
    assert cache.get("prompt1", "model1") is None
    cache.put("prompt1", "model1", "response1")
    assert cache.get("prompt1", "model1") == "response1"


def test_response_cache_ttl() -> None:
    cache = ResponseCache(ttl_seconds=0.01)
    cache.put("prompt", "model", "value")
    time.sleep(0.02)
    assert cache.get("prompt", "model") is None


def test_cached_embedder() -> None:
    base = LocalHashEmbedder(dims=32)
    cache = EmbeddingCache()
    cached = CachedEmbedder(base, cache)

    vecs1 = cached.embed(["hello world", "test"])
    assert cache.stats.misses == 2
    assert cache.stats.hits == 0

    vecs2 = cached.embed(["hello world", "test"])
    assert cache.stats.hits == 2
    assert vecs1 == vecs2


def test_cache_clear() -> None:
    cache = EmbeddingCache()
    cache.put("a", [1.0])
    assert cache.size == 1
    cache.clear()
    assert cache.size == 0
