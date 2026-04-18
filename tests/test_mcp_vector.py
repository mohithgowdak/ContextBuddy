from __future__ import annotations

from pathlib import Path

from contextbuddy.index.graph import RepoGraphIndex
from contextbuddy.index.vector import RepoVectorIndex


def _write(p: Path, text: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def test_vector_build_search_and_update(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _write(root / "a.py", "def hello_world():\n    return 1\n")
    _write(root / "b.md", "This project uses ContextEngine to compress context.\n")

    idx = RepoVectorIndex(root=root, index_dir=tmp_path / "indexes", embedder_id="localhash", embedder_config={"dims": 128})
    stats = idx.build(max_files=1000, batch_size=16)
    assert "index_path" in stats

    hits = idx.search("hello_world", top_k=5)
    assert len(hits) > 0
    assert any("hello_world" in h.preview for h in hits)

    _write(root / "a.py", "def goodbye_world():\n    return 2\n")
    upd = idx.update(max_files=1000, batch_size=16)
    assert upd["files_changed"] >= 1
    hits2 = idx.search("goodbye_world", top_k=5)
    assert any("goodbye_world" in h.preview for h in hits2)


def test_vector_search_prefers_src_segment_when_cosine_ties(tmp_path: Path) -> None:
    """Same chunk text in root vs under src/ gets identical embeddings; src should rank first with default boost."""
    root = tmp_path / "repo"
    body = "def foo():\n    return 42\n"
    _write(root / "b.py", body)
    _write(root / "src" / "a.py", body)

    idx = RepoVectorIndex(root=root, index_dir=tmp_path / "indexes", embedder_id="localhash", embedder_config={"dims": 128})
    idx.build(max_files=1000, batch_size=16)

    hits = idx.search("foo return", top_k=2, prefer_subpaths=["src"], prefer_subpath_boost=2.0)
    assert len(hits) >= 2
    assert "src" in hits[0].path.replace("\\", "/")


def test_vector_graph_hybrid_seed_expansion(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    _write(root / "pkg" / "a.py", "from .b import foo\n\ndef bar():\n    return foo()\n")
    _write(root / "pkg" / "b.py", "def foo():\n    return 123\n")

    idx_dir = tmp_path / "indexes"
    v = RepoVectorIndex(root=root, index_dir=idx_dir)
    v.build(max_files=1000, batch_size=16)
    v_hits = v.search("bar", top_k=5)
    assert v_hits

    g = RepoGraphIndex(root=root, index_dir=idx_dir)
    g.build(max_files=1000)
    g_matches = g.expand_from_files([m.path for m in v_hits], hop_limit=1, include_imports=True, top_k=10)
    names = {Path(m.path).name for m in g_matches}
    assert "b.py" in names

