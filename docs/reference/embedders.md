# Embedders (semantic upgrade path)

ContextBuddy is **embedder-agnostic**. Any object that implements:

```python
def embed(texts: list[str]) -> list[list[float]]:
    ...
```

can be used for scoring, retrieval, and caching.

## Levels (practical guidance)

- **Level 0 (default)**: `LocalHashEmbedder` (zero deps, deterministic, “semantic-ish” lexical fingerprint)
- **Level 1 (free semantic, local)**:
  - `OllamaEmbedder` (recommended DX; keeps Python deps light)
  - `SentenceTransformersEmbedder` (heavier install; fully in-process)
- **Level 2 (paid semantic)**:
  - `OpenAIEmbedder`
  - `GeminiEmbedder`

## Use with `ContextEngine`

```python
from contextbuddy import ContextEngine, ContextEngineConfig, OllamaEmbedder

engine = ContextEngine(
    ContextEngineConfig(max_context_tokens=4000),
    embedder=OllamaEmbedder(model="nomic-embed-text"),
)
```

## Use with `MemoryStore` / `Retriever`

```python
from contextbuddy import MemoryStore, Retriever, OllamaEmbedder

embedder = OllamaEmbedder(model="nomic-embed-text")
store = MemoryStore(embedder=embedder).add(chunks)
retriever = Retriever(store, embedder=embedder)
```

## Built-in embedders

All are defined in `src/contextbuddy/embedder.py`.

### `LocalHashEmbedder` (default)

- **Cost**: $0
- **Deps**: none
- **Best for**: docs where question and answer share vocabulary

### `OllamaEmbedder` (local, free semantic)

- **Cost**: $0 API tokens
- **Deps**: `httpx` (via `contextbuddy[ollama]`)
- **Requires**: Ollama installed + running (`http://localhost:11434`)
- **Endpoint**: `/api/embeddings`

### `SentenceTransformersEmbedder` (local, in-process)

- **Cost**: $0 API tokens
- **Deps**: `sentence-transformers` (pulls torch; heavier)

### `OpenAIEmbedder` (paid, best accuracy)

- **Cost**: API embeddings
- **Deps**: `openai` (via `contextbuddy[openai]`)

### `GeminiEmbedder` (paid, best accuracy)

- **Cost**: API embeddings
- **Deps**: `google-genai` (via `contextbuddy[gemini]`)

## Optional dependency behavior

All optional embedders raise `ImportError` with install instructions when dependencies are missing.

