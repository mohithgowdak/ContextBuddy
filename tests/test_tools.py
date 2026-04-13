import json

from contextbuddy.engine import ContextEngine, ContextEngineConfig
from contextbuddy.store.memory import MemoryStore
from contextbuddy.tools import make_search_tool, make_compress_tool, handle_tool_call


def _make_store() -> MemoryStore:
    store = MemoryStore()
    store.add([
        "Invoice INV-92831 issued on 2026-04-01.",
        "Weather forecast for next week shows rain.",
        "Support ticket ACME-2041 about chargebacks.",
    ], metadata={"source": "test.txt"})
    return store


def test_search_tool_schema() -> None:
    store = _make_store()
    tool = make_search_tool(store)
    assert tool["type"] == "function"
    assert tool["function"]["name"] == "search_documents"
    assert "query" in tool["function"]["parameters"]["properties"]


def test_search_tool_callable() -> None:
    store = _make_store()
    tool = make_search_tool(store)
    handler = tool["_callable"]
    result = handler(query="invoice")
    assert "INV-92831" in result
    assert "test.txt" in result


def test_compress_tool_schema() -> None:
    engine = ContextEngine(ContextEngineConfig(max_context_tokens=500))
    tool = make_compress_tool(engine)
    assert tool["type"] == "function"
    assert tool["function"]["name"] == "compress_context"


def test_compress_tool_callable() -> None:
    engine = ContextEngine(ContextEngineConfig(max_context_tokens=500))
    tool = make_compress_tool(engine)
    handler = tool["_callable"]
    result = handler(
        context="Long text about invoices. " * 50 + "\n\nUnrelated filler. " * 50,
        prompt="Summarize invoices",
    )
    data = json.loads(result)
    assert "compressed_prompt" in data
    assert data["final_tokens"] <= data["original_tokens"]


def test_handle_tool_call() -> None:
    store = _make_store()
    tools = [make_search_tool(store)]

    class FakeFunction:
        name = "search_documents"
        arguments = '{"query": "invoice"}'

    class FakeToolCall:
        function = FakeFunction()

    result = handle_tool_call(FakeToolCall(), tools)
    assert "INV-92831" in result


def test_handle_unknown_tool() -> None:
    result = handle_tool_call(
        {"function": {"name": "unknown", "arguments": "{}"}},
        [],
    )
    assert "error" in result
