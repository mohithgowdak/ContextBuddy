#!/usr/bin/env python3
"""
Stdio MCP entry for Cursor and other MCP hosts.

Resolves "Connection closed" when the host's `python` is not the same interpreter
where `pip install -e .` was run: we prepend this repo's `src/` to sys.path so
`import contextbuddy` works from a git checkout.

Still requires: pip install "contextbuddy[mcp]" (or pip install mcp) on that interpreter.
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if _SRC.is_dir():
    sys.path.insert(0, str(_SRC))

from contextbuddy.mcp.server import main  # noqa: E402

if __name__ == "__main__":
    main()
