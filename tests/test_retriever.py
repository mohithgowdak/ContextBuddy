from contextbuddy.retriever import Retriever
from contextbuddy.store.memory import MemoryStore


def _make_store() -> MemoryStore:
    store = MemoryStore()
    store.add([
        "Invoice INV-92831 issued on 2026-04-01 for account_id=acct_12345.",
        "Random text about weather and sports that is not relevant at all.",
        "Support ticket ACME-2041 mentions chargebacks for user_id=usr_9z8y7x6w.",
        "Company picnic planned for May 15th at the park with all employees.",
        "Machine learning model training completed with 95% accuracy on dataset.",
    ])
    return store


def test_retrieve_returns_chunks() -> None:
    store = _make_store()
    retriever = Retriever(store, top_k=3)
    chunks = retriever.retrieve("invoice and ticket IDs")
    assert isinstance(chunks, list)
    assert len(chunks) <= 3


def test_build_prompt() -> None:
    store = _make_store()
    retriever = Retriever(store, max_context_tokens=2000, dev_mode=False)
    prompt, report = retriever.build_prompt("What is the invoice number?")
    assert "User:" in prompt
    assert report.total_chunks > 0


def test_query() -> None:
    store = _make_store()
    retriever = Retriever(store, max_context_tokens=2000, dev_mode=False)
    result = retriever.query(
        "What is the invoice number?",
        llm_call=lambda p: f"ECHO: {p[:50]}",
    )
    assert result.startswith("ECHO:")
    assert retriever.last_report is not None


def test_last_search_results() -> None:
    store = _make_store()
    retriever = Retriever(store, top_k=5)
    retriever.retrieve("invoice")
    assert len(retriever.last_search_results) > 0
