# Pipeline (`Pipeline`)

`Pipeline` is the highest-level “one object” API that wires:

- loaders
- store
- retriever
- compressor (engine)
- optional router + caches

## Import

```python
from contextbuddy import Pipeline
```

## One-liner setup

```python
pipeline = Pipeline.from_directory("./docs/", dev_mode=True)
answer = pipeline.query("What are the payment terms?", llm_call=my_llm)
```

## Add more sources later

```python
pipeline.add("new.pdf")
pipeline.add(["a.pdf", "b.txt"])
pipeline.add("https://example.com/docs")
```

## Bring your own store/embedder

```python
from contextbuddy import Pipeline, MemoryStore, OllamaEmbedder

emb = OllamaEmbedder()
store = MemoryStore(embedder=emb)
pipeline = Pipeline(store=store, embedder=emb)
pipeline.add("./docs/")
```

