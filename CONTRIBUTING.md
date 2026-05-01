# Contributing to ContextBuddy

Thanks for helping improve ContextBuddy.

## Development setup

```bash
git clone [this repo]
cd ContextBuddy

python -m venv .venv
source .venv/bin/activate  # Windows: .\.venv\Scripts\Activate.ps1

pip install -e ".[dev]"
pytest -q
```

## What we optimize for

ContextBuddy is a context compression middleware. The product is **stripping efficiency**:
- maximize token reduction **without** losing answer-critical information
- preserve regex-matched entities **100%**
- keep chunk boundaries coherent (no mid-sentence cuts)

If a change improves token reduction but increases miss-rate, it’s a net loss.

## Red lines (must never regress)
- Regex-matched entities ALWAYS survive compression
- Compressed output is never empty for non-empty input
- No mid-sentence splits in final output
- Deterministic behavior for same input + config

Run:
```bash
pytest -q
pytest tests/test_redlines.py -v
```

## Benchmarks / quality gate

Run the bench harness:
```bash
python -m contextbuddy bench --gate --json bench-local.json
```

If you change chunking/scoring/entity rules, update/add benchmark cases and keep the gate green.

## Pull requests

Please include:
- what changed and why (1–3 bullets)
- test plan (`pytest` output or steps)
- any behavior changes (API, defaults, performance)

