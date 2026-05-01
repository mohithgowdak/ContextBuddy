from __future__ import annotations

from pathlib import Path

from contextbuddy.index.graph import RepoGraphIndex, build_default_index_dir


def _write(p: Path, text: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def test_graph_build_and_search_symbols(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _write(
        root / "pkg" / "a.py",
        "from .b import foo\n\n"
        "class Cat:\n"
        "    def meow(self):\n"
        "        return 1\n\n"
        "def bar():\n"
        "    return foo()\n",
    )
    _write(root / "pkg" / "b.py", "def foo():\n    return 2\n")

    idx_dir = tmp_path / "indexes"
    idx = RepoGraphIndex(root=root, index_dir=idx_dir)
    stats = idx.build(max_files=1000)
    assert "index_path" in stats

    matches = idx.search("Cat", top_k=10, hop_limit=1)
    assert any(m.kind == "symbol" and m.name == "Cat" for m in matches)

    # import expansion should pull b.py when searching for a.py related symbol
    matches2 = idx.search("bar", top_k=10, hop_limit=1)
    paths = {Path(m.path).name for m in matches2}
    assert "b.py" in paths


def test_graph_update_detects_changes(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _write(root / "x.py", "def old_name():\n    return 1\n")

    idx = RepoGraphIndex(root=root, index_dir=tmp_path / "indexes")
    idx.build(max_files=1000)
    assert any(m.name == "old_name" for m in idx.search("old_name", top_k=5))

    _write(root / "x.py", "def new_name():\n    return 1\n")
    upd = idx.update(max_files=1000)
    assert upd["files_changed"] >= 1
    assert any(m.name == "new_name" for m in idx.search("new_name", top_k=5))


def test_default_index_dir_is_path() -> None:
    p = build_default_index_dir()
    assert isinstance(p, Path)

