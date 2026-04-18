import types

import pytest


def test_ollama_embedder_import_error_when_httpx_missing(monkeypatch):
    # Force httpx import failure inside the adapter.
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "httpx":
            raise ImportError("no httpx")
        return real_import(name, *args, **kwargs)

    builtins.__import__ = fake_import
    try:
        from contextbuddy.embedder import OllamaEmbedder

        with pytest.raises(ImportError) as e:
            OllamaEmbedder()
        assert "contextbuddy[ollama]" in str(e.value)
    finally:
        builtins.__import__ = real_import


def test_sbert_embedder_import_error_when_missing(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "sentence_transformers":
            raise ImportError("no sbert")
        return real_import(name, *args, **kwargs)

    builtins.__import__ = fake_import
    try:
        from contextbuddy.embedder import SentenceTransformersEmbedder

        with pytest.raises(ImportError) as e:
            SentenceTransformersEmbedder()
        assert "contextbuddy[sbert]" in str(e.value)
    finally:
        builtins.__import__ = real_import


def test_gemini_embedder_import_error_when_missing(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        # google-genai is imported as `from google import genai`
        if name == "google":
            raise ImportError("no google-genai")
        return real_import(name, *args, **kwargs)

    builtins.__import__ = fake_import
    try:
        from contextbuddy.embedder import GeminiEmbedder

        with pytest.raises(ImportError) as e:
            GeminiEmbedder()
        assert "contextbuddy[gemini]" in str(e.value)
    finally:
        builtins.__import__ = real_import


def test_ollama_embedder_parses_response_without_network():
    from contextbuddy.embedder import OllamaEmbedder

    class DummyResp:
        def raise_for_status(self):
            return None

        def json(self):
            return {"embedding": [0.1, 0.2, 0.3]}

    class DummyClient:
        def post(self, path, json):
            assert path == "/api/embeddings"
            assert "model" in json and "prompt" in json
            return DummyResp()

    e = OllamaEmbedder(client=DummyClient())
    vecs = e.embed(["hello", "world"])
    assert vecs == [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]


def test_gemini_embedder_parses_response_without_network():
    from contextbuddy.embedder import GeminiEmbedder

    class DummyEmbedding:
        values = [0.5, 0.25]

    class DummyResp:
        embedding = DummyEmbedding()

    class DummyModels:
        def embed_content(self, model, contents):
            assert model
            assert isinstance(contents, str)
            return DummyResp()

    class DummyClient:
        models = DummyModels()

    # The adapter enforces an import guard in __init__. Only run this test if
    # `google-genai` is actually installed (i.e., `from google import genai` works).
    try:
        from google import genai  # type: ignore  # noqa: F401
    except Exception:
        pytest.skip("google-genai not installed in test environment")

    e = GeminiEmbedder(client=DummyClient())
    vecs = e.embed(["a", "b"])
    assert vecs == [[0.5, 0.25], [0.5, 0.25]]

