# Stores (`MemoryStore`, `PersistentStore`)

Stores hold chunk text + vectors + metadata, and provide similarity search for RAG.

## `MemoryStore`

Defined in `src/contextbuddy/store/memory.py`.

- In-memory, zero-dep
- Uses an embedder (default `LocalHashEmbedder`) + cosine similarity
- Good for <100k chunks

```python
from contextbuddy import MemoryStore

store = MemoryStore().add(chunks, metadata={"source": "docs/"})
results = store.search("payment terms", top_k=10)
```

## `PersistentStore`

Defined in `src/contextbuddy/store/persistent.py`.

- Wraps `MemoryStore`
- Saves/loads a JSON file

```python
from contextbuddy import PersistentStore

store = PersistentStore("./index.json")
store.add(chunks)
```

## Metadata

Metadata is attached at `add()` time and returned on search results.

