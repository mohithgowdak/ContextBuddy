# CLI (`python -m contextbuddy`)

ContextBuddy ships a CLI for demos and smoke tests.

## Compress

```bash
python -m contextbuddy compress --prompt "What are the key points?" --file report.txt --show-prompt
```

Notes:

- `--file` currently reads **text files**.
- For PDFs/DOCX/URLs, use `load()` in Python (see `docs/reference/loaders.md`).

## Bench (quality gate)

```bash
python -m contextbuddy bench --gate --json bench-report.json
```

