from __future__ import annotations

import asyncio
import hashlib
import math
import re
from dataclasses import dataclass
from typing import List, Sequence

from .types import Embedder


_word_re = re.compile(r"[A-Za-z0-9_]{2,}")


def _token_hash(token: str) -> int:
    """
    Deterministic, process-independent token hash.

    Python's built-in `hash()` is randomized per interpreter process
    (via PYTHONHASHSEED), which would make `LocalHashEmbedder` produce
    different vectors across runs — violating the determinism red line.
    `hashlib.md5` is stable everywhere.
    """
    return int.from_bytes(hashlib.md5(token.encode("utf-8")).digest()[:8], "big")


def _l2_normalize(vec: List[float]) -> List[float]:
    n2 = sum(v * v for v in vec)
    if n2 <= 0.0:
        return vec
    inv = 1.0 / math.sqrt(n2)
    return [v * inv for v in vec]


@dataclass(frozen=True)
class LocalHashEmbedder:
    """
    Offline, dependency-free embedder using a hashing trick over word tokens.

    It is not a true semantic embedding model, but it provides a surprisingly
    useful relevance signal for paragraph pruning and keeps the library
    zero-friction out of the box.

    Deterministic across processes (uses `hashlib.md5`, not `hash()`).
    """

    dims: int = 256

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        out: List[List[float]] = []
        dims = int(self.dims)
        if dims <= 0:
            raise ValueError("dims must be > 0")

        for text in texts:
            vec = [0.0] * dims
            for m in _word_re.finditer(text.lower()):
                token = m.group(0)
                h = _token_hash(token)
                idx = h % dims
                sign = -1.0 if (h & 1) else 1.0
                vec[idx] += sign
            out.append(_l2_normalize(vec))
        return out

    async def aembed(self, texts: Sequence[str]) -> List[List[float]]:
        return await asyncio.to_thread(self.embed, texts)


class OpenAIEmbedder:
    """
    Optional embedder using the `openai` Python SDK.

    Install: `pip install "contextbuddy[openai]"`

    Pass any OpenAI-compatible client via `client=` to reuse your own
    retry/timeout configuration. Pass an `AsyncOpenAI` instance to get
    a true-async `aembed`; otherwise `aembed` falls back to running the
    sync path in a worker thread.
    """

    def __init__(self, *, model: str = "text-embedding-3-small", client=None):
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as e:
            raise ImportError(
                "OpenAIEmbedder requires the 'openai' package. "
                "Install with: pip install \"contextbuddy[openai]\""
            ) from e

        self._model = model
        self._client = client or OpenAI()

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        resp = self._client.embeddings.create(model=self._model, input=list(texts))
        return [d.embedding for d in resp.data]

    async def aembed(self, texts: Sequence[str]) -> List[List[float]]:
        create = getattr(getattr(self._client, "embeddings", None), "create", None)
        if create is not None and asyncio.iscoroutinefunction(create):
            resp = await create(model=self._model, input=list(texts))
            return [d.embedding for d in resp.data]
        return await asyncio.to_thread(self.embed, texts)


class OllamaEmbedder:
    """
    Optional embedder using a local Ollama server (free, offline embeddings).

    Install: `pip install "contextbuddy[ollama]"`
    Requires Ollama running locally: https://ollama.com/

    Notes:
    - Keeps ContextBuddy's Python deps light (no torch).
    - Uses HTTP to talk to the local Ollama daemon.
    """

    def __init__(
        self,
        *,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
        timeout: float = 30.0,
        client=None,
    ):
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._client = client  # optional injected client (sync or async)
        self._httpx = None

        # Only require httpx when we need to construct a real HTTP client.
        # If the caller injects a client (e.g. for tests), we can operate without httpx.
        if self._client is None:
            try:
                import httpx  # type: ignore
            except ImportError as e:
                raise ImportError(
                    "OllamaEmbedder requires 'httpx'. Install with: pip install \"contextbuddy[ollama]\""
                ) from e
            self._httpx = httpx

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        # Sync client
        if self._client is None:
            # _httpx is guaranteed when _client is None (see __init__)
            with self._httpx.Client(base_url=self._base_url, timeout=self._timeout) as c:
                return self._embed_with_sync_client(c, texts)
        return self._embed_with_sync_client(self._client, texts)

    def _embed_with_sync_client(self, client, texts: Sequence[str]) -> List[List[float]]:
        out: List[List[float]] = []
        for t in texts:
            r = client.post("/api/embeddings", json={"model": self._model, "prompt": str(t)})
            r.raise_for_status()
            data = r.json()
            vec = data.get("embedding")
            if not isinstance(vec, list):
                raise RuntimeError("OllamaEmbedder: invalid response (missing embedding)")
            out.append([float(x) for x in vec])
        return out

    async def aembed(self, texts: Sequence[str]) -> List[List[float]]:
        # If caller injected an AsyncClient, use it; otherwise use httpx.AsyncClient.
        if self._client is None:
            # _httpx is guaranteed when _client is None (see __init__)
            async with self._httpx.AsyncClient(base_url=self._base_url, timeout=self._timeout) as c:
                return await self._embed_with_async_client(c, texts)

        post = getattr(self._client, "post", None)
        if post is not None and asyncio.iscoroutinefunction(post):
            return await self._embed_with_async_client(self._client, texts)
        return await asyncio.to_thread(self.embed, texts)

    async def _embed_with_async_client(self, client, texts: Sequence[str]) -> List[List[float]]:
        out: List[List[float]] = []
        for t in texts:
            r = await client.post("/api/embeddings", json={"model": self._model, "prompt": str(t)})
            r.raise_for_status()
            data = r.json()
            vec = data.get("embedding")
            if not isinstance(vec, list):
                raise RuntimeError("OllamaEmbedder: invalid response (missing embedding)")
            out.append([float(x) for x in vec])
        return out


class SentenceTransformersEmbedder:
    """
    Optional embedder using `sentence-transformers` (local, in-process embeddings).

    Install: `pip install "contextbuddy[sbert]"`
    Heavier install (pulls torch), but runs fully offline once installed.
    """

    def __init__(self, *, model: str = "sentence-transformers/all-MiniLM-L6-v2", device: str | None = None):
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as e:
            raise ImportError(
                "SentenceTransformersEmbedder requires 'sentence-transformers'. "
                "Install with: pip install \"contextbuddy[sbert]\""
            ) from e

        self._model_name = model
        self._device = device
        self._model = SentenceTransformer(model, device=device) if device else SentenceTransformer(model)

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        vecs = self._model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
        # sentence-transformers may return numpy arrays; convert to plain lists.
        return [list(map(float, v.tolist() if hasattr(v, "tolist") else v)) for v in vecs]

    async def aembed(self, texts: Sequence[str]) -> List[List[float]]:
        return await asyncio.to_thread(self.embed, texts)


class GeminiEmbedder:
    """
    Optional embedder using Google's Gemini embeddings via `google-genai`.

    Install: `pip install "contextbuddy[gemini]"`
    Requires API key configuration for Google GenAI.
    """

    def __init__(self, *, model: str = "text-embedding-004", client=None):
        try:
            from google import genai  # type: ignore
        except ImportError as e:
            raise ImportError(
                "GeminiEmbedder requires 'google-genai'. Install with: pip install \"contextbuddy[gemini]\""
            ) from e

        self._model = model
        self._client = client or genai.Client()

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        out: List[List[float]] = []
        for t in texts:
            resp = self._client.models.embed_content(model=self._model, contents=str(t))
            emb = getattr(resp, "embedding", None)
            vec = getattr(emb, "values", None) if emb is not None else None
            if vec is None:
                raise RuntimeError("GeminiEmbedder: invalid response (missing embedding values)")
            out.append([float(x) for x in vec])
        return out

    async def aembed(self, texts: Sequence[str]) -> List[List[float]]:
        # google-genai client is sync; run in thread.
        return await asyncio.to_thread(self.embed, texts)
