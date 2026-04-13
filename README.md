<p align="center">
  <h1 align="center">ContextBuddy</h1>
  <p align="center">
    <strong>From raw PDFs to compressed prompts in 3 lines. Cut your LLM bill by 60%.</strong>
  </p>
</p>

<p align="center">
  <a href="https://pypi.org/project/contextbuddy/"><img src="https://img.shields.io/pypi/v/contextbuddy?color=blue" alt="PyPI"></a>
  <a href="https://pypi.org/project/contextbuddy/"><img src="https://img.shields.io/pypi/pyversions/contextbuddy" alt="Python"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/yourname/contextbuddy" alt="License"></a>
</p>

<!-- TODO: Replace with actual GIF recording of the CLI demo -->
```
┌──────────────────────────────────────────────────────┐
│                   ContextBuddy                       │
├──────────────────────────────────────────────────────┤
│  Tokens   before    15000   after      3000          │
│  Saved    -80.0%              Est. $0.0600           │
│  [████████████████████████░░░░░░] 12000 tokens freed │
├──────────────────────────────────────────────────────┤
│  Chunks   total 12    kept 4    pruned 8             │
│  Entities INV-92831, 2026-04-01, acct_12345          │
└──────────────────────────────────────────────────────┘
```

---

## Why ContextBuddy?

| Feature | LangChain | LlamaIndex | LightRAG | **ContextBuddy** |
|---------|-----------|------------|----------|-|
| Install size | 100+ deps | 50+ deps | 20+ deps | **0 core deps** |
| Lines to first RAG | ~30 | ~15 | ~10 | **3** |
| Cost optimization | None | None | None | **Built-in** |
| ROI telemetry | None | None | None | **Every call** |
| Vector DB required | Yes | Yes | Yes | **No** |
| Context compression | None | None | None | **Semantic pruning + budgeting** |
| PDF/URL/DOCX loading | Separate install | Built-in | Separate | **Built-in (optional deps)** |

**ContextBuddy does 80% of what LangChain does in 10% of the code.** Zero dependencies for the core. Optional extras for PDFs, web scraping, and accurate tokenizers.

---

## Install

```bash
pip install contextbuddy
```

Optional extras:

```bash
pip install "contextbuddy[pdf]"         # PDF loading (pymupdf)
pip install "contextbuddy[web]"         # URL/web scraping (httpx + bs4)
pip install "contextbuddy[tiktoken]"    # Accurate OpenAI token counts
pip install "contextbuddy[openai]"      # OpenAI embeddings
pip install "contextbuddy[loaders]"     # All document loaders
pip install "contextbuddy[all]"         # Everything
```

---

## 4 Ways to Use It (pick your level)

### Path 1: Compress raw text (3 lines)

```python
from contextbuddy import ContextEngine, ContextEngineConfig

engine = ContextEngine(ContextEngineConfig(dev_mode=True, max_context_tokens=4000))
result = engine.run(
    user_prompt="Summarize the key points.",
    context=huge_raw_text,
    llm_call=lambda p: client.responses.create(model="gpt-4o-mini", input=p),
)
```

### Path 2: Load files + compress (3 lines)

```python
from contextbuddy import ContextEngine, load

engine = ContextEngine(dev_mode=True, max_context_tokens=4000)
result = engine.run(
    user_prompt="What are the payment terms?",
    context=load("contract.pdf"),
    llm_call=lambda p: client.responses.create(model="gpt-4o-mini", input=p),
)
```

### Path 3: Multi-document RAG (3 lines)

```python
from contextbuddy import Retriever, MemoryStore, load

store = MemoryStore().add(load("./docs/"))
result = Retriever(store, dev_mode=True).query(
    "What are the payment terms?",
    llm_call=lambda p: client.responses.create(model="gpt-4o-mini", input=p),
)
```

### Path 4: Full pipeline (one-liner setup)

```python
from contextbuddy import Pipeline

pipeline = Pipeline.from_directory("./docs/", dev_mode=True)
result = pipeline.query("Summarize the contract", llm_call=my_llm)
```

---

## Architecture

```
Your Files (PDFs, URLs, DOCX, TXT, CSV, directories)
    │
    ▼
┌─────────────┐
│  Loaders    │  load("file.pdf") / load("https://...") / load("./dir/")
└─────┬───────┘
      │
      ▼
┌─────────────┐
│  Store      │  In-memory vector index (auto-dedup, metadata, persistence)
└─────┬───────┘
      │
      ▼
┌─────────────┐
│  Retriever  │  Semantic search → top-k chunks
└─────┬───────┘
      │
      ▼
┌─────────────┐
│  Compressor │  Prune → entity keep-list → token budget → compose
└─────┬───────┘
      │
      ▼
┌─────────────┐
│  Router     │  Score query complexity → pick cheap or expensive model
└─────┬───────┘
      │
      ▼
┌─────────────┐
│  Cache      │  Embedding cache + response cache (skip redundant work)
└─────┬───────┘
      │
      ▼
  Your LLM (OpenAI / Anthropic / Google / Local)
```

Every layer is optional. Use one, use all, or use any combination.

---

## Document Loaders

```python
from contextbuddy.loaders import load

load("report.pdf")                    # PDF (pip install contextbuddy[pdf])
load("https://docs.example.com")      # Web page (pip install contextbuddy[web])
load("notes.docx")                    # Word doc (pip install contextbuddy[docx])
load("data.csv")                      # CSV (rows as chunks)
load("config.json")                   # JSON (keys/items as chunks)
load("./documents/")                  # Entire directory (recursive)
load(["a.pdf", "b.txt", "c.docx"])   # Batch load
```

Zero-dep formats: `.txt`, `.md`, `.csv`, `.json`, `.log`, `.xml`, `.yaml`, `.html`

---

## Vector Store

```python
from contextbuddy import MemoryStore, PersistentStore, load

# In-memory (default)
store = MemoryStore()
store.add(load("report.pdf"), metadata={"source": "report.pdf"})
store.add(load("notes.txt"), metadata={"source": "notes.txt"})
results = store.search("payment terms", top_k=10)

# Persistent (survives restarts)
store = PersistentStore("./my_index.json")
store.add(load("./docs/"))
# Auto-saves to disk. Reloads on next init.
```

Features: auto-deduplication, metadata tracking, serialization, pure-Python cosine search.

---

## Smart Model Router

Route simple queries to cheap models. Route complex ones to expensive models. All offline.

```python
from contextbuddy import Router, Pipeline

router = Router([
    {"max_complexity": 0.3, "model": "gpt-4o-mini"},
    {"max_complexity": 1.0, "model": "gpt-4o"},
])

pipeline = Pipeline.from_directory("./docs/", router=router, dev_mode=True)
result = pipeline.query(
    "Summarize the contract",
    llm_calls={
        "gpt-4o-mini": lambda p: cheap_client.responses.create(model="gpt-4o-mini", input=p),
        "gpt-4o": lambda p: expensive_client.responses.create(model="gpt-4o", input=p),
    },
)
```

---

## Caching

```python
from contextbuddy import Pipeline, EmbeddingCache, ResponseCache

pipeline = Pipeline.from_directory(
    "./docs/",
    embedding_cache=EmbeddingCache(persist_path="./cache/embeddings.json"),
    response_cache=ResponseCache(ttl_seconds=3600),
)
# First query embeds + calls LLM. Second identical query: instant.
```

---

## Agent Tools

ContextBuddy generates OpenAI-compatible function/tool schemas for agents:

```python
from contextbuddy.tools import make_search_tool, make_compress_tool, handle_tool_call

tools = [make_search_tool(store), make_compress_tool(engine)]

response = client.chat.completions.create(
    model="gpt-4o",
    messages=messages,
    tools=tools,
)

# Dispatch tool calls
for tc in response.choices[0].message.tool_calls:
    result = handle_tool_call(tc, tools)
```

---

## Streaming

```python
for chunk in engine.run(
    user_prompt="Summarize",
    context=load("report.pdf"),
    llm_call=lambda p: client.responses.create(model="gpt-4o-mini", input=p, stream=True),
    stream=True,
):
    print(chunk, end="")
```

---

## OpenAI Drop-in Wrapper

Zero code changes to your existing app:

```python
from contextbuddy import wrap_openai

client = wrap_openai(openai.OpenAI(), max_context_tokens=4000, dev_mode=True)
# Use client.chat.completions.create() exactly as before.
# System messages are automatically compressed.
```

---

## CLI (no API key needed)

```bash
echo "Your huge context..." | python -m contextbuddy compress \
    --prompt "What are the key points?" \
    --max-tokens 2000 \
    --show-prompt

python -m contextbuddy compress \
    --file report.txt \
    --prompt "Extract action items" \
    --model gpt-4o
```

---

## Entity Types Preserved

| Category | Examples |
|----------|----------|
| Emails | `alice@example.com` |
| URLs | `https://api.example.com/v2/users` |
| Dates | `2026-04-13`, `04/13/2026`, `2026-04-13T10:30` |
| UUIDs | `550e8400-e29b-41d4-a716-446655440000` |
| Tickets | `JIRA-1234`, `ACME-2041` |
| Phone numbers | `+1-555-867-5309` |
| Money | `$4,500.00`, `1000 USD` |
| IPs | `192.168.1.100` |
| ID-like values | `account_id=acct_12345` |
| Versions | `v2.1.0` |

---

## Pre-built Model Pricing

```python
from contextbuddy.pricing import (
    OPENAI_GPT4O, OPENAI_GPT4O_MINI, OPENAI_GPT41, OPENAI_GPT41_MINI,
    OPENAI_O3, OPENAI_O4_MINI,
    CLAUDE_OPUS_4, CLAUDE_SONNET_4, CLAUDE_HAIKU_35,
    GEMINI_25_PRO, GEMINI_25_FLASH,
    LOCAL_FREE,
    get_pricing,  # get_pricing("gpt-4o") -> ModelPricing
)
```

---

## Programmatic Report

```python
report = engine.last_report

report.original_prompt_tokens   # 15000
report.final_prompt_tokens      # 3000
report.reduction_pct            # 80.0
report.estimated_savings        # 0.06
report.kept_chunks              # 4
report.total_chunks             # 12
report.entities                 # ["INV-92831", "2026-04-01", ...]
```

---

## FAQ

**Will this hurt answer quality?**
It can if you prune too aggressively. Start with `min_relevance=0.10` and inspect the compressed prompt in dev mode. The entity keep-list ensures critical data points survive.

**Does it send my data anywhere?**
Not by default. The built-in embedder and vector store run 100% locally with zero dependencies. Only if you explicitly plug in `OpenAIEmbedder` does it call an external API.

**Does it work with streaming?**
Yes. Pass `stream=True` to `engine.run()`. ContextBuddy emits the ROI report, then yields LLM chunks.

**How accurate is the token count?**
The default `HeuristicTokenizer` uses a 4-chars-per-token rule. For exact counts: `pip install contextbuddy[tiktoken]`.

**Can I use this in production?**
Yes. The core pipeline is deterministic, dependency-free, and fast (<10ms for typical payloads). Set `dev_mode=False` to disable telemetry.

**How is this different from LangChain?**
ContextBuddy is **compression-first**. LangChain retrieves context but sends it all to the LLM. ContextBuddy retrieves, compresses, preserves entities, and shows you exactly how much you're saving. Zero core dependencies vs 100+.

---

## Contributing

```bash
git clone https://github.com/yourname/contextbuddy.git
cd contextbuddy
pip install -e ".[dev]"
pytest
```

---

## License

MIT License. See [LICENSE](LICENSE).
