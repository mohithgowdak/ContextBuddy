from __future__ import annotations

from pathlib import Path

import pytest


def _write(p: Path, text: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def test_codegraph_build_extracts_calls_when_optional_deps_installed(tmp_path: Path) -> None:
    pytest.importorskip("tree_sitter")
    pytest.importorskip("tree_sitter_languages")

    from contextbuddy.index.codegraph import RepoCodeGraphIndex

    root = tmp_path / "repo"
    root.mkdir()
    _write(
        root / "a.py",
        "def foo():\n"
        "    return 1\n"
        "\n"
        "def bar():\n"
        "    return foo()\n",
    )

    idx = RepoCodeGraphIndex(root=root, index_dir=tmp_path / "indexes")
    stats = idx.build(max_files=1000)
    assert "index_path" in stats

    edges = idx.top_calls_for_paths(["a.py"], limit=50)
    assert any(e.caller.endswith("bar") and "foo" in e.callee for e in edges)

