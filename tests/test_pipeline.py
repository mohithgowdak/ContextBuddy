import os
import tempfile
from pathlib import Path

from contextbuddy.pipeline import Pipeline
from contextbuddy.store.memory import MemoryStore


def test_pipeline_add_and_query() -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write("Invoice INV-92831 issued on 2026-04-01 for acct_12345.\n\n")
        f.write("Unrelated text about weather and sports activities.\n\n")
        f.write("Support ticket ACME-2041 for user_id=usr_9z8y7x6w.\n\n")
        f.flush()
        path = f.name

    try:
        pipeline = Pipeline(dev_mode=False, max_context_tokens=500)
        pipeline.add(path)

        assert pipeline.store.size > 0

        result = pipeline.query(
            "What is the invoice number?",
            llm_call=lambda p: f"ECHO: {p[:80]}",
        )
        assert "ECHO:" in result
        assert pipeline.last_report is not None
    finally:
        os.unlink(path)


def test_pipeline_from_directory() -> None:
    with tempfile.TemporaryDirectory() as d:
        p1 = Path(d) / "doc1.txt"
        p1.write_text("Machine learning algorithms and neural networks.", encoding="utf-8")
        p2 = Path(d) / "doc2.txt"
        p2.write_text("Cooking recipes for Italian pasta with fresh ingredients.", encoding="utf-8")

        pipeline = Pipeline.from_directory(d, dev_mode=False, max_context_tokens=500)
        assert pipeline.store.size >= 2

        result = pipeline.query(
            "Tell me about machine learning",
            llm_call=lambda p: "OK",
        )
        assert result == "OK"


def test_pipeline_from_sources() -> None:
    with tempfile.TemporaryDirectory() as d:
        p1 = Path(d) / "a.txt"
        p1.write_text("Document A about payments and invoices.", encoding="utf-8")
        p2 = Path(d) / "b.txt"
        p2.write_text("Document B about customer support tickets.", encoding="utf-8")

        pipeline = Pipeline.from_sources([str(p1), str(p2)], dev_mode=False)
        assert pipeline.store.size >= 2


def test_pipeline_chaining() -> None:
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "test.txt"
        p.write_text("Test content here about technology.", encoding="utf-8")

        pipeline = Pipeline(dev_mode=False)
        pipeline.add(str(p))
        assert pipeline.store.size >= 1


def test_pipeline_with_response_cache() -> None:
    from contextbuddy.cache import ResponseCache

    cache = ResponseCache()
    pipeline = Pipeline(dev_mode=False, response_cache=cache)
    pipeline.store.add(["Some content about invoices and billing."])

    call_count = 0

    def counting_llm(p: str) -> str:
        nonlocal call_count
        call_count += 1
        return "response"

    pipeline.query("test question", llm_call=counting_llm)
    assert call_count == 1

    pipeline.query("test question", llm_call=counting_llm)
    assert call_count == 1  # cached
    assert cache.stats.hits == 1
