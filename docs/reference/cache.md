# Caching (`EmbeddingCache`, `ResponseCache`)

ContextBuddy includes two small caches to reduce repeat work.

## `EmbeddingCache`

- Stores embeddings for identical text
- Optional persistence to JSON file

```python
from contextbuddy import EmbeddingCache, CachedEmbedder, OllamaEmbedder

base = OllamaEmbedder()
cache = EmbeddingCache(persist_path="./cache/embeddings.json")
embedder = CachedEmbedder(base, cache)
```

## `ResponseCache`

- Stores LLM responses by `(prompt, model)` with TTL

```python
from contextbuddy import ResponseCache

cache = ResponseCache(ttl_seconds=3600)
```

## Pipeline usage

```python
from contextbuddy import Pipeline, EmbeddingCache, ResponseCache

pipeline = Pipeline.from_directory(
    "./docs/",
    embedding_cache=EmbeddingCache(persist_path="./cache/embeddings.json"),
    response_cache=ResponseCache(ttl_seconds=3600),
)
```

