# Loaders (`load()`)

ContextBuddy supports loading context from multiple sources into chunk lists.

## Import

```python
from contextbuddy import load
```

## Supported inputs

- **Files**: `.txt`, `.md`, `.csv`, `.json`, `.log`, `.xml`, `.yaml`, `.html`
- **PDF**: requires `contextbuddy[pdf]`
- **DOCX**: requires `contextbuddy[docx]`
- **URLs**: requires `contextbuddy[web]`
- **Directories**: recursively loads supported files
- **Batches**: list of paths/urls

## Examples

```python
chunks = load("report.pdf")
chunks = load("notes.docx")
chunks = load("https://docs.example.com/page")
chunks = load("./docs/")               # directory
chunks = load(["a.pdf", "b.txt"])      # batch
```

## Optional dependency behavior

Each loader is guarded with an explicit `ImportError` telling you what to install.

## Implementation map

- `src/contextbuddy/loaders/__init__.py`: dispatcher
- `src/contextbuddy/loaders/text.py`: text/CSV/JSON
- `src/contextbuddy/loaders/pdf.py`: PyMuPDF extraction + SmartChunker(pdf)
- `src/contextbuddy/loaders/web.py`: httpx + BeautifulSoup extraction
- `src/contextbuddy/loaders/docx.py`: python-docx extraction
- `src/contextbuddy/loaders/directory.py`: recursive directory loader

