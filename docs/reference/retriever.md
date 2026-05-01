# Retriever (`Retriever`)

`Retriever` is the “multi-doc RAG in one object” layer:

**search → compress → call LLM**

It wires:

- `MemoryStore` search (top-k chunks)
- `ContextEngine` compression (budget + entities)
- your `llm_call`

## Import

```python
from contextbuddy import Retriever, MemoryStore
```

## Usage

```python
store = MemoryStore().add(chunks)
retriever = Retriever(store, max_context_tokens=3000, top_k=20, dev_mode=True)

answer = retriever.query(
    "What are the payment terms?",
    llm_call=lambda prompt: client.responses.create(model="gpt-4o-mini", input=prompt),
)
```

## Async

```python
answer = await retriever.aquery(
    "What are the payment terms?",
    llm_call=lambda prompt: async_client.responses.create(model="gpt-4o-mini", input=prompt),
)
```

## Using a semantic embedder

Pass the same embedder to both the store and the retriever:

```python
from contextbuddy import MemoryStore, Retriever, OllamaEmbedder

emb = OllamaEmbedder(model="nomic-embed-text")
store = MemoryStore(embedder=emb).add(chunks)
retriever = Retriever(store, embedder=emb)
```

