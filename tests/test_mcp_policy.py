from __future__ import annotations

import pytest

from contextbuddy.mcp.policy import (
    MCPInputError,
    max_prompt_chars,
    stub_report_dict,
    validate_context,
    validate_prompt,
    validate_query_str,
)


def test_validate_prompt_too_large(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CONTEXTBUDDY_MCP_MAX_PROMPT_CHARS", "10")
    assert max_prompt_chars() == 10
    with pytest.raises(MCPInputError) as ei:
        validate_prompt("x" * 11)
    assert ei.value.code == "INPUT_TOO_LARGE"


def test_validate_context_total(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CONTEXTBUDDY_MCP_MAX_CONTEXT_CHARS", "20")
    with pytest.raises(MCPInputError):
        validate_context("a" * 21)


def test_stub_report_keys() -> None:
    s = stub_report_dict()
    assert "original_prompt_tokens" in s and "entities" in s


def test_validate_query(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CONTEXTBUDDY_MCP_MAX_QUERY_CHARS", "5")
    with pytest.raises(MCPInputError):
        validate_query_str("abcdef")
