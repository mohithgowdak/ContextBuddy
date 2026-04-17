# Benchmarks

ContextBuddy ships a small benchmark harness to prevent “more compression” changes from silently breaking correctness.

## What we measure

- **Answer survival rate**: proxy metric — does the compressed prompt still contain the answer-bearing substring?
- **Entity survival rate**: hard guarantee — regex-detected entities must be present in the compressed output **and** the report.
- **Compression reduction %**: token reduction on the final prompt.
- **Latency**: mean and p95 for compression.

## Run (quality gate)

```bash
python -m pip install -e .
python -m contextbuddy bench --gate --json bench-report.json
```

## Custom dataset

Provide a JSON file:

```bash
python -m contextbuddy bench --dataset benchmarks/datasets/v0.json --max-tokens 800 --gate --json bench-v0.json
```

Dataset format:

```json
[
  {
    "name": "support/invoice-id",
    "document": "....big text....",
    "question": "What is the invoice ID and date?",
    "expected_substring": "INV-92831 issued 2026-04-01",
    "required_entities": ["INV-92831", "2026-04-01", "acct_12345"]
  }
]
```

