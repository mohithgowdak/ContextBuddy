<p align="center">
  <h1 align="center">ContextBuddy</h1>
  <p align="center">
    <strong>From raw PDFs to compressed prompts in 3 lines. Cut your LLM bill by 60%.</strong>
  </p>
</p>

<p align="center">
  <a href="https://github.com/mohithgowdak/ContextBuddy"><img src="https://img.shields.io/github/stars/mohithgowdak/ContextBuddy?style=social" alt="Stars"></a>
  <img src="https://img.shields.io/badge/version-0.3.0-blue" alt="Version">
  <img src="https://img.shields.io/badge/python-3.9%2B-blue" alt="Python">
  <a href="LICENSE"><img src="https://img.shields.io/github/license/mohithgowdak/ContextBuddy" alt="License"></a>
  <img src="https://img.shields.io/badge/dependencies-0_(core)-brightgreen" alt="Deps">
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

## What is ContextBuddy?

ContextBuddy is a **lightweight, open-source Python library** that acts as a context middleware between your raw data (PDFs, web pages, documents, databases) and your LLM call. Its entire job is to take a massive, messy prompt -- like 20 pages of scraped text -- compress it, filter out the noise, preserve critical entities, and pass a clean, token-efficient prompt to any LLM.

Think of it as the **missing layer** in every AI stack: the part that makes sure you're not paying for 15,000 tokens when only 3,000 actually matter.

---

## The Problem

Every developer building with LLMs hits the same wall:

1. **You're overpaying.** You send 15,000 tokens of scraped text to GPT-4 when only 3,000 tokens actually matter. That's 5x the cost for worse results.
2. **Context is noisy.** Raw PDFs, web scrapes, and database dumps are full of irrelevant paragraphs, boilerplate, and filler. Your LLM wastes attention on noise.
3. **Critical details get lost.** When you manually truncate context to save tokens, you accidentally drop the one invoice ID or date the user asked about.
4. **Existing frameworks are bloated.** LangChain has 100+ dependencies. LlamaIndex has 50+. You just want to load a PDF and ask a question.

---

## The Solution

ContextBuddy solves all four problems in a single library:

- **Semantic pruning** -- scores every paragraph against your question and drops irrelevant content before it hits the expensive model.
- **Entity preservation** -- automatically extracts IDs, dates, URLs, phone numbers, and other critical data points, ensuring they are never accidentally pruned.
- **Token budgeting** -- enforces a strict token limit so your context always fits the window you set.
- **ROI telemetry** -- prints exactly how many tokens (and dollars) you saved on every call. Developers screenshot this and share it.

It works with **any LLM** -- OpenAI, Anthropic, Google, or local models -- because it only touches the prompt, not the model.

---

## Who is this for?

ContextBuddy works at every scale. The value just shows up differently:

| Scale | How they use it | Why it matters |
|-------|----------------|----------------|
| **Solo dev / hobbyist** | Drop-in middleware, skip LangChain entirely | Zero deps, 3 lines, no infrastructure to manage |
| **Startup (seed to Series A)** | Full pipeline replacing LangChain stack | Cut API bill from $10k to $3k/month, ship in days not weeks |
| **Mid-size company** | Compression layer inside their existing LangChain/LlamaIndex stack | Bolt on to existing code, save 60% without rewriting anything |
| **Enterprise** | Cost governance + smart routing across teams | ROI telemetry for budgeting, model routing to manage spend at scale |

The bigger the company, the more they overpay on tokens. A team running 1M LLM calls/day is burning $30k+/month in unnecessary tokens. A compression middleware that saves 60% is worth $18k/month to them -- and it plugs in with 3 lines.

Specifically built for:

- **AI engineers** building RAG pipelines who want to cut API costs without sacrificing answer quality.
- **Startups** shipping LLM-powered products who need to keep their OpenAI/Anthropic bill under control.
- **Solo developers** who want multi-doc RAG without installing LangChain and 100 transitive dependencies.
- **Platform teams** who need cost visibility and governance over LLM spend across the organization.
- **Agent builders** who need their tools to pass compressed, high-signal context to function calls.
- **Anyone already using LangChain/LlamaIndex** who wants to cut costs without rewriting -- just drop ContextBuddy into your existing pipeline as a compression step.

---

## Why ContextBuddy over the alternatives?

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

## 90-second quickstart (the only path you need)

Compress a huge, noisy context into a budgeted prompt before the LLM call — in three lines.

```python
from contextbuddy import ContextEngine, ContextEngineConfig

engine = ContextEngine(ContextEngineConfig(dev_mode=True, max_context_tokens=4000))

huge_context = """
Invoice INV-92831 issued 2026-04-01 for account_id=acct_12345.
Amount: $4,500.00 USD. Payment due within 30 days.

... 20 pages of unrelated notes, meeting transcripts, old emails ...

Ticket ACME-2041: chargebacks for user_id=usr_9z8y7x6w.
"""

final_prompt, report = engine.build_prompt(
    user_prompt="Summarize the invoice and ticket. Include all IDs and dates.",
    context=huge_context,
)

print(report.reduction_pct, "% smaller,  $", report.estimated_savings, "saved per call")
# Pass `final_prompt` to any LLM (OpenAI, Anthropic, Gemini, local — ContextBuddy doesn't care).
```

When you're ready to call an LLM, use `engine.run(...)` (sync) or `engine.arun(...)` (async) and pass any `llm_call` callable. See [4 Ways to Use It](#4-ways-to-use-it-pick-your-level) for loaders, full RAG, and pipeline patterns.

---

## Benchmarks (quality gate)

ContextBuddy includes a small benchmark harness so “more compression” doesn’t silently break correctness.

```bash
python -m pip install -e .
python -m contextbuddy bench --gate --json bench-report.json
```

See `docs/benchmarks.md` and `benchmarks/datasets/v0.sample.json`.

---

## What ContextBuddy guarantees

- **Entity survival.** Any regex-matched entity (IDs, emails, URLs, dates, money, tickets, phones, UUIDs, version strings) always survives compression.
- **Never larger.** Output is always shorter than input — or unchanged if input already fits the budget.
- **Never empty.** If input has content, output is non-empty. Empty output is treated as a bug, not a valid result.
- **Deterministic core.** Same input + same config = same output. No randomness in the core pipeline.
- **Zero core dependencies.** Works on a fresh Python 3.9+ install. `pip install contextbuddy` → done.
- **Budget respected.** Final prompt always fits `max_context_tokens`. No mid-sentence cuts.

## What ContextBuddy does not do

- **Not an agent framework.** It compresses context; it doesn't orchestrate tools, memory, or loops. Pair with LangGraph/CrewAI if you need that.
- **Not a vector database.** The in-memory store is great up to ~100k chunks. Above that, use Pinecone/Weaviate and plug ContextBuddy in as the compression layer.
- **Doesn't call LLMs itself.** You always pass `llm_call=...`. Works with OpenAI, Anthropic, Gemini, Ollama, anything.
- **Doesn't learn.** Scoring is algorithmic (BM25 + stemmer + synonyms + n-grams). No training, no drift.
- **Doesn't ship a UI.** It's a library, not a product.

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

## How Compression Actually Works (No ML, No NumPy)

ContextBuddy doesn't use a neural network to compress your context. The entire pipeline is algorithmic, using techniques that predate deep learning by decades -- but combined in a way that delivers results competitive with embedding-based approaches. Here's exactly what happens when you call `engine.run()`:

### Step 1: Chunking

Your raw text (from a PDF, web scrape, or string) is split into paragraphs using regex on double newlines. Tiny fragments under 40 characters are dropped as noise.

### Step 2: Relevance Scoring (HybridScorer -- the secret sauce)

This is where ContextBuddy is different from every other compression library. Instead of relying on a single signal, the default `HybridScorer` combines **four independent scoring signals** into one relevance score:

**Signal 1: BM25 (70% weight)** -- The same algorithm that powers Elasticsearch and Lucene. It handles term-frequency saturation (saying "payment" 10 times isn't 10x more relevant than once), document-length normalization (longer paragraphs don't cheat the ranking), and inverse-document-frequency weighting (rare words matter more than common ones). This alone is a massive upgrade over naive keyword matching.

**Signal 2: Stemming (built into BM25)** -- A lightweight suffix-stripping stemmer normalizes word forms before scoring. "payments" matches "payment". "running" matches "run". "organized" matches "organizing". No NLTK, no spaCy -- just 120 lines of pure Python implementing the most impactful Porter stemmer rules.

**Signal 3: Synonym Expansion (15% weight)** -- A built-in thesaurus of ~200 word groups covering business, legal, tech, medical, and general vocabulary. When you ask about "car insurance," the scorer automatically expands "car" to also check for "automobile," "vehicle," and "auto" in every paragraph. "Buy" matches "purchase." "Salary" matches "compensation." "Error" matches "bug." All offline, zero API calls.

**Signal 4: Character N-gram Fuzzy Matching (15% weight)** -- Catches morphological variants and typos that stemming misses. "optimise" matches "optimize." "colour" matches "color." Works by computing Jaccard similarity over character trigrams -- if two words share enough 3-character substrings, they're treated as partial matches.

The four signals are normalized to [0, 1] and combined with configurable weights. The result: paragraphs that are genuinely relevant to your question score high, even when they use completely different words.

```python
from contextbuddy import HybridScorer

scorer = HybridScorer()
scores = scorer.score(
    query="What is the car insurance policy?",
    chunks=[
        "The automobile coverage plan includes collision and liability.",  # scores HIGH (synonym match)
        "Employee cafeteria hours are 12pm to 2pm.",                      # scores LOW (irrelevant)
    ],
)
```

### Step 3: Entity Extraction

Regex patterns scan every paragraph for critical data: emails, URLs, dates, dollar amounts, IDs, phone numbers, ticket numbers, etc. Any paragraph containing a detected entity is **force-kept** regardless of its relevance score, so you never accidentally drop the invoice ID the user asked about.

### Step 4: Budget Enforcement

The surviving paragraphs are sorted by importance (entity-containing chunks first, then by relevance score) and greedily packed into the token budget. If even a single chunk won't fit, it's extractively summarized (leading sentences kept until the limit). The final prompt always fits the budget you set.

### Scorer Comparison

| | `HybridScorer` (default) | `SemanticScorer` + `LocalHashEmbedder` | `SemanticScorer` + `OpenAIEmbedder` |
|--|--|--|--|
| Understands synonyms | **Yes** (built-in thesaurus) | No | Yes |
| Handles word forms | **Yes** (stemming) | No | Yes |
| Fuzzy matching | **Yes** (n-grams) | No | No |
| IDF weighting | **Yes** (BM25) | No | Yes |
| Needs API key | **No** | No | Yes |
| Needs internet | **No** | No | Yes |
| Dependencies | **Zero** | Zero | `openai` package |
| Cost | **Free** | Free | ~$0.0002/doc |
| Latency | **<5ms** | <2ms | ~200ms |

The `HybridScorer` is the default because it gives the best results for zero cost and zero dependencies. For production use cases with highly specialized vocabulary (niche medical terms, non-English content), you can still swap in `OpenAIEmbedder` for true neural semantic matching:

```python
from contextbuddy.embedder import OpenAIEmbedder

engine = ContextEngine(
    ContextEngineConfig(max_context_tokens=4000, dev_mode=True),
    embedder=OpenAIEmbedder(),  # neural embeddings for edge cases
)
```

Or bring your own scorer -- any object with a `score(query=..., chunks=...) -> List[float]` method works.

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

## Real-World Use Cases

### Customer Support Bot

Your chatbot pulls a customer's full history (invoices, tickets, emails, notes) for every query -- ~15,000 tokens. Most of it is irrelevant.

```python
from contextbuddy import Pipeline

pipeline = Pipeline.from_directory("./customer_data/acct_12345/", dev_mode=True, max_context_tokens=3000)
answer = pipeline.query("What was my last invoice amount?", llm_call=my_llm)
# [ContextBuddy] 15000 → 2800 tokens (81.3% reduction). Est. savings: $0.0305
# Entity keep-list preserved: INV-92831, $4,500.00, 2026-04-01, acct_12345
```

At 10,000 queries/day: **$11,250/month without ContextBuddy vs $2,250/month with it.**

### Legal Document Review

A law firm uploads a 50-page contract. Lawyers ask questions about specific clauses.

```python
from contextbuddy import Pipeline

pipeline = Pipeline.from_directory("./contracts/", dev_mode=True, max_context_tokens=4000)
answer = pipeline.query("What are the payment terms and late penalties?", llm_call=my_llm)
```

ContextBuddy loads the PDF, indexes 200+ paragraphs, retrieves the relevant ones, prunes to the 5 that matter, and preserves all clause numbers, dates, and dollar amounts. Without it, you'd need LangChain + a vector database + 50 lines of glue code.

### Internal Knowledge Base

500 internal docs (Confluence exports, PDFs, Markdown). Engineers ask questions via Slack bot.

```python
from contextbuddy import Pipeline, PersistentStore, Router

pipeline = Pipeline(
    store=PersistentStore("./index.json"),
    router=Router([
        {"max_complexity": 0.3, "model": "gpt-4o-mini"},
        {"max_complexity": 1.0, "model": "gpt-4o"},
    ]),
    dev_mode=True,
)
pipeline.add("./company_docs/")
answer = pipeline.query(slack_message, llm_calls={"gpt-4o-mini": cheap_fn, "gpt-4o": expensive_fn})
```

Simple questions ("What's the WiFi password?") route to the cheap model. Complex questions ("Compare our auth architecture options") route to the expensive one. **Router alone saves 60-70% on top of compression.**

---

## When NOT to Use ContextBuddy

Being honest:

- **Full agent orchestration** (multi-step reasoning, tool chains, long-term memory) -- use LangGraph or CrewAI instead. ContextBuddy compresses context, it doesn't orchestrate agents.
- **Billion-scale vector search** -- if you have 100M+ documents and need sub-millisecond search, use Pinecone or Weaviate directly. ContextBuddy's in-memory store is designed for <100k chunks.
- **Already deep in LangChain and it's working** -- don't rewrite. But you *can* drop ContextBuddy inside your existing LangChain pipeline as a compression step:

```python
from contextbuddy import ContextEngine

engine = ContextEngine(max_context_tokens=4000)

# Inside your LangChain chain, after retrieval but before the LLM call:
compressed_prompt, report = engine.build_prompt(
    user_prompt=user_question,
    context=retrieved_documents,   # from your existing LangChain retriever
)
# Pass compressed_prompt to your LLM instead of the raw retrieved docs
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

## Why I Built This

I'm a Recent CS Grad. I was deep in the rabbit hole of context engineering -- reading papers, watching talks, experimenting with how LLMs actually use the context you feed them. And I kept hitting the same wall.

I had a project that needed RAG. Load some PDFs, ask questions, get answers. Simple, right? So I reached for LangChain. And then I spent two days wrestling with 100+ dependencies, cryptic abstractions, and a codebase that felt like it was designed for a different problem. I just wanted to load a PDF and compress the context before sending it to an LLM. I didn't need an agent framework. I didn't need a plugin ecosystem. I needed maybe 200 lines of focused code.

So I closed my laptop, went for a walk, and thought: **what if the entire layer between "raw data" and "LLM call" was just... simple?**

That's what ContextBuddy is. It's the library I wished existed when I started.

The core insight was that most LLM applications are sending 5-10x more context than they need to. You scrape a 50-page contract, dump the whole thing into GPT-4, and pay for 15,000 tokens when only 3,000 matter. The LLM doesn't even perform better with the extra noise -- it performs *worse*. Context engineering isn't about stuffing more tokens in. It's about sending the *right* tokens.

I built ContextBuddy with a few principles:

1. **Zero dependencies for the core.** If you just want to compress text, you shouldn't need to install anything else. No numpy. No torch. No tiktoken. Just Python.
2. **Three lines to integrate.** If it takes more than that, developers will bounce. I know because I bounced.
3. **Show the ROI.** Every call prints exactly how many tokens and dollars you saved. Not because it's a gimmick -- because developers need to justify tool choices to their managers, and a screenshot of "$0.12 saved per call" does that instantly.
4. **Grow with you.** Start with 3 lines. When you need PDF loading, add it. When you need a vector store, add it. When you need model routing, add it. You should never have to rip out ContextBuddy and replace it with LangChain because you outgrew it.

I'm not claiming this replaces LangChain for every use case. If you need multi-step agent orchestration with tool chains and long-term memory, LangChain/LangGraph is the right call. But for the 80% of LLM applications that just need to load data, compress context, and call a model? ContextBuddy does it in a fraction of the code, with zero bloat, and it shows you exactly how much money you're saving.

This started as a side project born out of frustration. I'm sharing it because I think every developer building with LLMs deserves a simpler option.

If it saves you time or money, star the repo. That's all I ask.

---

## Contributing

```bash
git clone https://github.com/mohithgowdak/ContextBuddy.git
cd contextbuddy
pip install -e ".[dev]"
pytest
```

---

## License

MIT License. See [LICENSE](LICENSE).
