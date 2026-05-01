"""
Indexing subsystem (optional).

This is intentionally lightweight and stdlib-only:
- a persistent repo graph (imports + Python symbol spans)
- query-time retrieval that selects minimal context

MCP tools can use this to avoid scanning the whole repo on every query.
"""

from .graph import (
    GraphMatch,
    RepoGraphIndex,
    build_default_index_dir,
)
from .vector import RepoVectorIndex, VectorMatch, make_embedder

__all__ = [
    "RepoGraphIndex",
    "GraphMatch",
    "build_default_index_dir",
    "RepoVectorIndex",
    "VectorMatch",
    "make_embedder",
]

