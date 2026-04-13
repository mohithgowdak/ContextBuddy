<p align="center">
  <h1 align="center">ContextBuddy</h1>
  <p align="center">
    <strong>Drop this into your code. Cut your LLM token bill by 60%+. Zero config.</strong>
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

## The Problem

You're sending **15,000 tokens** of scraped text to GPT-4 when only **3,000 tokens** actually matter. You're paying 5x more than you need to, and your responses are slower and noisier.

## The Solution

**ContextBuddy** is a lightweight Python middleware that sits between your raw context and your LLM call. It:

1. **Semantically prunes** irrelevant paragraphs (cheap local scoring, no API calls)
2. **Preserves key entities** so IDs, dates, URLs, and ticket numbers are never lost
3. **Enforces a token budget** so your context always fits the window you set
4. **Prints ROI telemetry** so you can see (and screenshot) exactly how much you're saving

It works with **any model** — OpenAI, Anthropic, Google, local Llama — because it only touches the prompt, not the model.

---

## Install

```bash
pip install contextbuddy
```

Optional extras for production use:

```bash
pip install "contextbuddy[tiktoken]"    # Accurate OpenAI token counts
pip install "contextbuddy[openai]"      # OpenAI embeddings for better pruning
pip install "contextbuddy[all]"         # Everything
```

---

## 3-Line Integration

### Option A: Engine wrapper (works with any LLM)

```python
from contextbuddy import ContextEngine, ContextEngineConfig

engine = ContextEngine(ContextEngineConfig(dev_mode=True, max_context_tokens=4000))
result = engine.run(
    user_prompt="Summarize the key points.",
    context=huge_raw_text,
    llm_call=lambda prompt: client.responses.create(model="gpt-4o-mini", input=prompt),
)
```

### Option B: OpenAI drop-in wrapper (zero code changes)

```python
from contextbuddy import wrap_openai

client = wrap_openai(openai.OpenAI(), max_context_tokens=4000, dev_mode=True)
# Use client.chat.completions.create() exactly as before.
# System messages are automatically compressed.
```

### Option C: Async support

```python
result = await engine.arun(
    user_prompt="Summarize this.",
    context=huge_text,
    llm_call=lambda prompt: async_client.responses.create(model="gpt-4o-mini", input=prompt),
)
```

---

## CLI (instant demo, no API key needed)

```bash
echo "Your huge context here..." | python -m contextbuddy compress \
    --prompt "What are the key points?" \
    --max-tokens 2000 \
    --show-prompt
```

Or from a file:

```bash
python -m contextbuddy compress \
    --file scraped_page.txt \
    --prompt "Extract action items" \
    --max-tokens 1500 \
    --model gpt-4o
```

---

## How It Works

```
Raw Context (15k tokens)
    │
    ▼
┌─────────────┐
│  Chunking   │  Split into paragraphs
└─────┬───────┘
      │
      ▼
┌─────────────┐
│  Scoring    │  Embed + cosine similarity vs. user prompt
└─────┬───────┘
      │
      ▼
┌─────────────┐
│  Pruning    │  Drop chunks below min_relevance threshold
└─────┬───────┘
      │
      ▼
┌─────────────┐
│  Entities   │  Extract IDs, dates, URLs, etc. → keep-list guardrail
└─────┬───────┘
      │
      ▼
┌─────────────┐
│  Budgeting  │  Enforce max_context_tokens (drop tail → summarize)
└─────┬───────┘
      │
      ▼
  Compressed Prompt (3k tokens) → Your LLM
```

### Entity types detected

| Category | Examples |
|----------|----------|
| Emails | `alice@example.com` |
| URLs | `https://api.example.com/v2/users` |
| Dates | `2026-04-13`, `04/13/2026` |
| UUIDs | `550e8400-e29b-41d4-a716-446655440000` |
| Tickets | `JIRA-1234`, `ACME-2041` |
| Phone numbers | `+1-555-867-5309` |
| Money | `$4,500.00`, `€99.99`, `1000 USD` |
| IPs | `192.168.1.100` |
| ID-like values | `account_id=acct_12345`, `ref=TXN-789` |
| Versions | `v2.1.0` |

---

## Configuration Reference

```python
from contextbuddy import ContextEngine, ContextEngineConfig
from contextbuddy.pricing import OPENAI_GPT4O, CLAUDE_SONNET_4

engine = ContextEngine(
    ContextEngineConfig(
        max_context_tokens=4000,    # Hard token budget for context
        min_relevance=0.15,         # Cosine similarity threshold (0.0–1.0)
        dev_mode=True,              # Print ROI telemetry to stderr
        rich_output=True,           # Box-drawing colored output (vs. one-liner)
        pricing=OPENAI_GPT4O,       # Model pricing for cost estimates
        include_entities_section=True,  # Inject KeyEntities: section
        chunk_min_chars=40,         # Min chars per chunk
    ),
)
```

### Pre-built pricing presets

```python
from contextbuddy.pricing import (
    OPENAI_GPT4O, OPENAI_GPT4O_MINI, OPENAI_GPT41, OPENAI_GPT41_MINI,
    OPENAI_O3, OPENAI_O4_MINI,
    CLAUDE_OPUS_4, CLAUDE_SONNET_4, CLAUDE_HAIKU_35,
    GEMINI_25_PRO, GEMINI_25_FLASH,
    LOCAL_FREE,
    get_pricing,  # get_pricing("gpt-4o") → ModelPricing
)
```

### Custom embedder / tokenizer

```python
from contextbuddy.embedder import OpenAIEmbedder
from contextbuddy.tokenizer import TiktokenTokenizer

engine = ContextEngine(
    ContextEngineConfig(dev_mode=True),
    embedder=OpenAIEmbedder(model="text-embedding-3-small"),
    tokenizer=TiktokenTokenizer(encoding_name="cl100k_base"),
)
```

---

## Programmatic Report

Every call exposes `engine.last_report` (or the second return value of `build_prompt`):

```python
final_prompt, report = engine.build_prompt(user_prompt=..., context=...)

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
Not by default. The built-in embedder (`LocalHashEmbedder`) runs 100% locally with zero dependencies. Only if you explicitly plug in `OpenAIEmbedder` does it call an external API.

**Does it work with streaming?**
Yes. ContextBuddy compresses the prompt *before* your LLM call. Streaming, tool use, and function calling all work normally.

**How accurate is the token count?**
The default `HeuristicTokenizer` uses a 4-chars-per-token rule (surprisingly accurate for English). For exact counts, install `tiktoken`: `pip install contextbuddy[tiktoken]`.

**Can I use this in production?**
Yes. The core pipeline is deterministic, dependency-free, and fast (<10ms for typical payloads). Set `dev_mode=False` to disable telemetry output.

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
