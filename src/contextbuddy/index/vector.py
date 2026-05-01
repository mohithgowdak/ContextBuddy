from __future__ import annotations

import json
import math
import os
import subprocess
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from ..chunking import SmartChunker
from ..embedder import LocalHashEmbedder
from ..types import Embedder
from .graph import DEFAULT_GRAPH_EXCLUDE_DIRS, DEFAULT_GRAPH_EXTS, _iter_files, _file_fingerprint, _safe_read_text, _sha1, _now_iso


def _git_head(root: Path) -> Optional[str]:
    try:
        if not (root / ".git").exists():
            return None
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
        head = (res.stdout or "").strip()
        return head or None
    except Exception:
        return None


@dataclass(frozen=True)
class VectorRecord:
    id: str
    path: str  # repo-relative path
    chunk_hash: str
    text: str
    vector: List[float]
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class VectorMatch:
    path: str
    score: float
    id: str
    preview: str
    metadata: Dict[str, Any]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += float(x) * float(y)
        na += float(x) * float(x)
        nb += float(y) * float(y)
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / math.sqrt(na * nb)


def _embedder_fingerprint(embedder_id: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    # Minimal, stable fingerprint so users can keep multiple indexes.
    model = str(cfg.get("model") or cfg.get("name") or "")
    dims = int(cfg.get("dims") or 0)
    return {
        "id": str(embedder_id),
        "model": model,
        "dims": dims,
        "fingerprint": _sha1(f"{embedder_id}:{model}:{dims}"),
    }


def make_embedder(embedder_id: str, *, config: Optional[Dict[str, Any]] = None) -> Embedder:
    """
    Lazy embedder factory for MCP indexing.

    `embedder_id` values:
      - localhash
      - ollama
      - sbert
      - openai
      - gemini
    """
    cfg = dict(config or {})
    eid = (embedder_id or "localhash").strip().lower()
    if eid in ("localhash", "hash", "default"):
        dims = int(cfg.get("dims") or 256)
        return LocalHashEmbedder(dims=dims)
    if eid == "ollama":
        from ..embedder import OllamaEmbedder

        return OllamaEmbedder(
            model=str(cfg.get("model") or "nomic-embed-text"),
            base_url=str(cfg.get("base_url") or "http://localhost:11434"),
            timeout=float(cfg.get("timeout") or 30.0),
        )
    if eid in ("sbert", "sentence-transformers", "sentence_transformers"):
        from ..embedder import SentenceTransformersEmbedder

        return SentenceTransformersEmbedder(
            model=str(cfg.get("model") or "sentence-transformers/all-MiniLM-L6-v2"),
            device=cfg.get("device"),
        )
    if eid == "openai":
        from ..embedder import OpenAIEmbedder

        return OpenAIEmbedder(model=str(cfg.get("model") or "text-embedding-3-small"))
    if eid == "gemini":
        from ..embedder import GeminiEmbedder

        return GeminiEmbedder(model=str(cfg.get("model") or "text-embedding-004"))
    raise ValueError(f"Unknown embedder_id: {embedder_id}")


class RepoVectorIndex:
    """
    Persistent vector index over repo chunks (stdlib-only persistence).

    This is intentionally not a "vector database". It's a small, file-backed index
    designed for IDE/MCP usage:
    - build once
    - update incrementally on file changes
    - search quickly without scanning the whole repo every time
    """

    SCHEMA_VERSION = 1

    def __init__(
        self,
        *,
        root: str | Path,
        index_dir: str | Path,
        embedder_id: str = "localhash",
        embedder_config: Optional[Dict[str, Any]] = None,
        exclude_dirs: Optional[Sequence[str]] = None,
        exts: Optional[Sequence[str]] = None,
        chunk_min_chars: int = 100,
        chunk_merge_under_chars: int = 200,
    ) -> None:
        self.root = Path(root).resolve()
        self.index_dir = Path(index_dir).resolve()
        self.exclude_dirs = list(exclude_dirs) if exclude_dirs is not None else sorted(DEFAULT_GRAPH_EXCLUDE_DIRS)
        self.exts = list(exts) if exts is not None else sorted(DEFAULT_GRAPH_EXTS)
        self.embedder_id = (embedder_id or "localhash").strip().lower()
        self.embedder_config = dict(embedder_config or {})

        self._index_path = self.index_dir / _sha1(str(self.root))
        self._path = self._index_path / "vectors.json"

        self._chunker = SmartChunker(min_chars=int(chunk_min_chars), merge_under_chars=int(chunk_merge_under_chars))
        self._embedder: Optional[Embedder] = None

        self._data: Dict[str, Any] = {}

    def exists(self) -> bool:
        return self._path.exists() and self._path.stat().st_size > 0

    def _get_embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = make_embedder(self.embedder_id, config=self.embedder_config)
        return self._embedder

    def load(self) -> None:
        if not self.exists():
            raise FileNotFoundError(f"Vector index not found at {self._path}")
        self._data = json.loads(self._path.read_text(encoding="utf-8"))

    def build(
        self,
        *,
        max_files: int = 50_000,
        max_bytes_per_file: int = 512_000,
        batch_size: int = 64,
    ) -> Dict[str, Any]:
        if not self.root.exists() or not self.root.is_dir():
            raise FileNotFoundError(f"Repo root not found or not a directory: {self.root}")
        self._index_path.mkdir(parents=True, exist_ok=True)

        embedder = self._get_embedder()
        records: List[VectorRecord] = []
        fingerprints: Dict[str, Dict[str, Any]] = {}

        buf_texts: List[str] = []
        buf_meta: List[Dict[str, Any]] = []
        buf_rel: List[str] = []
        buf_hash: List[str] = []

        def flush() -> None:
            nonlocal records
            if not buf_texts:
                return
            vecs = embedder.embed(list(buf_texts))
            if len(vecs) != len(buf_texts):
                raise RuntimeError(
                    f"Embedder returned {len(vecs)} vectors for {len(buf_texts)} chunks"
                )
            for rel, chash, txt, meta, vec in zip(buf_rel, buf_hash, buf_texts, buf_meta, vecs):
                rid = _sha1(f"{rel}:{chash}")
                records.append(VectorRecord(id=rid, path=rel, chunk_hash=chash, text=txt, vector=[float(x) for x in vec], metadata=meta))
            buf_texts.clear()
            buf_meta.clear()
            buf_rel.clear()
            buf_hash.clear()

        chunk_count = 0
        file_count = 0

        for fp in _iter_files(self.root, exts=self.exts, exclude_dirs=self.exclude_dirs, max_files=int(max_files)):
            file_count += 1
            rel = str(fp.relative_to(self.root))
            fingerprints[rel] = _file_fingerprint(fp)
            text = _safe_read_text(fp, max_bytes=int(max_bytes_per_file))
            if text is None:
                continue

            doc_type = "code" if fp.suffix.lower() == ".py" else "auto"
            chunks = self._chunker.chunk(text, doc_type=doc_type)
            for i, ch in enumerate(chunks):
                ch = (ch or "").strip()
                if not ch:
                    continue
                chash = _sha1(ch)
                meta = {"path": rel, "chunk_index": int(i), "doc_type": doc_type}
                buf_texts.append(ch)
                buf_meta.append(meta)
                buf_rel.append(rel)
                buf_hash.append(chash)
                chunk_count += 1
                if len(buf_texts) >= int(batch_size):
                    flush()
        flush()

        self._data = {
            "schema_version": self.SCHEMA_VERSION,
            "root": str(self.root),
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
            "git_head": _git_head(self.root),
            "exclude_dirs": list(self.exclude_dirs),
            "exts": list(self.exts),
            "chunker": {
                "min_chars": int(self._chunker.min_chars),
                "merge_under_chars": int(self._chunker.merge_under_chars),
            },
            "embedder": _embedder_fingerprint(self.embedder_id, self.embedder_config),
            "files": fingerprints,
            "records": [asdict(r) for r in records],
            "stats": {"files_seen": int(file_count), "chunks": int(chunk_count)},
        }
        self._save()
        return {"index_path": str(self._path), **self._data.get("stats", {}), "updated_at": self._data["updated_at"]}

    def update(
        self,
        *,
        max_files: int = 50_000,
        max_bytes_per_file: int = 512_000,
        batch_size: int = 64,
        prune_deleted: bool = True,
    ) -> Dict[str, Any]:
        if not self.exists():
            return self.build(max_files=max_files, max_bytes_per_file=max_bytes_per_file, batch_size=batch_size)
        self.load()

        old_files: Dict[str, Dict[str, Any]] = dict(self._data.get("files", {}))
        new_files: Dict[str, Dict[str, Any]] = {}
        changed: Set[str] = set()
        seen: Set[str] = set()

        for fp in _iter_files(self.root, exts=self.exts, exclude_dirs=self.exclude_dirs, max_files=int(max_files)):
            rel = str(fp.relative_to(self.root))
            seen.add(rel)
            new_fp = _file_fingerprint(fp)
            new_files[rel] = new_fp
            if old_files.get(rel) != new_fp:
                changed.add(rel)

        deleted = set(old_files.keys()) - seen
        if not prune_deleted:
            deleted = set()

        keep_paths = set(old_files.keys()) - changed - deleted
        old_records = list(self._data.get("records", []))
        kept: List[Dict[str, Any]] = [r for r in old_records if r.get("path") in keep_paths]

        embedder = self._get_embedder()
        new_records: List[VectorRecord] = []

        buf_texts: List[str] = []
        buf_meta: List[Dict[str, Any]] = []
        buf_rel: List[str] = []
        buf_hash: List[str] = []

        def flush() -> None:
            nonlocal new_records
            if not buf_texts:
                return
            vecs = embedder.embed(list(buf_texts))
            if len(vecs) != len(buf_texts):
                raise RuntimeError(
                    f"Embedder returned {len(vecs)} vectors for {len(buf_texts)} chunks"
                )
            for rel, chash, txt, meta, vec in zip(buf_rel, buf_hash, buf_texts, buf_meta, vecs):
                rid = _sha1(f"{rel}:{chash}")
                new_records.append(VectorRecord(id=rid, path=rel, chunk_hash=chash, text=txt, vector=[float(x) for x in vec], metadata=meta))
            buf_texts.clear()
            buf_meta.clear()
            buf_rel.clear()
            buf_hash.clear()

        added_chunks = 0
        for rel in sorted(changed):
            abs_fp = (self.root / rel).resolve()
            text = _safe_read_text(abs_fp, max_bytes=int(max_bytes_per_file))
            if text is None:
                continue
            doc_type = "code" if abs_fp.suffix.lower() == ".py" else "auto"
            chunks = self._chunker.chunk(text, doc_type=doc_type)
            for i, ch in enumerate(chunks):
                ch = (ch or "").strip()
                if not ch:
                    continue
                chash = _sha1(ch)
                meta = {"path": rel, "chunk_index": int(i), "doc_type": doc_type}
                buf_texts.append(ch)
                buf_meta.append(meta)
                buf_rel.append(rel)
                buf_hash.append(chash)
                added_chunks += 1
                if len(buf_texts) >= int(batch_size):
                    flush()
        flush()

        combined = kept + [asdict(r) for r in new_records]
        self._data["updated_at"] = _now_iso()
        self._data["git_head"] = _git_head(self.root)
        self._data["files"] = new_files
        self._data["records"] = combined
        self._data["stats"] = {"files_seen": int(len(new_files)), "chunks": int(len(combined))}
        self._save()
        return {
            "index_path": str(self._path),
            "files_scanned": int(len(new_files)),
            "files_changed": int(len(changed)),
            "files_deleted": int(len(deleted)),
            "chunks_added": int(added_chunks),
            "chunks_total": int(len(combined)),
            "updated_at": self._data["updated_at"],
        }

    def search(self, query: str, *, top_k: int = 20, min_score: float = 0.0, max_preview_chars: int = 900) -> List[VectorMatch]:
        q = (query or "").strip()
        if not q:
            return []
        if not self.exists():
            raise FileNotFoundError(f"Vector index not found at {self._path}")
        if not self._data:
            self.load()
        embedder = self._get_embedder()

        q_vec = embedder.embed([q])[0]
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for r in self._data.get("records", []):
            vec = r.get("vector") or []
            s = _cosine(q_vec, vec)
            if s >= float(min_score):
                scored.append((float(s), r))
        scored.sort(key=lambda kv: kv[0], reverse=True)

        out: List[VectorMatch] = []
        for s, r in scored[: int(top_k)]:
            txt = str(r.get("text") or "")
            prev = txt.strip()
            if len(prev) > int(max_preview_chars):
                prev = prev[: int(max_preview_chars)].rstrip() + "\n..."
            out.append(
                VectorMatch(
                    path=str((self.root / str(r.get("path") or "")).resolve()),
                    score=float(s),
                    id=str(r.get("id") or ""),
                    preview=prev,
                    metadata=dict(r.get("metadata") or {}),
                )
            )
        return out

    def matches_to_context(self, matches: Sequence[VectorMatch]) -> List[str]:
        chunks: List[str] = []
        for m in matches:
            body = (m.preview or "").strip()
            if not body:
                continue
            chunks.append(f"Source: {m.path}\n{body}".strip())
        return chunks

    def _save(self) -> None:
        self._index_path.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self._data, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self._path)

