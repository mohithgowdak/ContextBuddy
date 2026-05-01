from __future__ import annotations

from contextbuddy.chunking import SmartChunker


def test_code_chunker_keeps_functions_intact() -> None:
    code = """
import os

def foo(x: int) -> int:
    y = x + 1
    return y

def bar() -> str:
    s = "hello"
    return s
""".strip()

    chunks = SmartChunker().chunk(code, doc_type="code")
    joined = "\n\n".join(chunks)

    # Both defs must exist in output, and not be split into tiny fragments.
    assert "def foo" in joined
    assert "return y" in joined
    assert "def bar" in joined
    assert "return s" in joined
    assert all(len(c) >= 50 for c in chunks)

