from __future__ import annotations

from contextbuddy.retriever import rrf_fuse


def test_rrf_fuse_prefers_items_appearing_in_multiple_lists() -> None:
    a = ["x", "y", "z"]
    b = ["y", "x", "w"]
    fused = rrf_fuse([a, b], k=60)
    keys = [k for k, _ in fused]
    assert keys[0] in {"x", "y"}
    assert "z" in keys
    assert "w" in keys

