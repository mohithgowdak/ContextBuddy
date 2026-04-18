"""
MCP policy: input caps, optional execution timeouts, and stable error payloads.

No logging of user content — use structured error codes only.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from typing import Any, Callable, Dict, List, Optional, Sequence, TypeVar, Union

T = TypeVar("T")


def _env_int(name: str, default: int) -> int:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return max(0, int(raw))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        return max(0.0, float(raw))
    except ValueError:
        return default


def max_prompt_chars() -> int:
    return _env_int("CONTEXTBUDDY_MCP_MAX_PROMPT_CHARS", 100_000)


def max_context_total_chars() -> int:
    return _env_int("CONTEXTBUDDY_MCP_MAX_CONTEXT_CHARS", 5_000_000)


def max_query_chars() -> int:
    return _env_int("CONTEXTBUDDY_MCP_MAX_QUERY_CHARS", 20_000)


def tool_timeout_sec() -> float:
    return _env_float("CONTEXTBUDDY_MCP_TOOL_TIMEOUT_SEC", 120.0)


def index_timeout_sec() -> float:
    return _env_float("CONTEXTBUDDY_MCP_INDEX_TIMEOUT_SEC", 600.0)


def stub_report_dict() -> Dict[str, Any]:
    """Matches ContextReport fields (for error paths that must still return a report)."""
    return {
        "original_prompt_tokens": 0,
        "final_prompt_tokens": 0,
        "original_context_tokens": 0,
        "final_context_tokens": 0,
        "reduction_pct": 0.0,
        "estimated_savings": 0.0,
        "kept_chunks": 0,
        "total_chunks": 0,
        "entities": [],
        "selected_indices": [],
    }


class MCPInputError(Exception):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(message)


class MCPToolTimeout(Exception):
    def __init__(self, seconds: float) -> None:
        self.seconds = seconds
        super().__init__(f"Tool exceeded timeout ({seconds}s)")


def context_total_chars(context: Union[str, Sequence[str]]) -> int:
    if isinstance(context, str):
        return len(context)
    return sum(len(str(x)) for x in context)


def validate_prompt(user_prompt: str, *, field: str = "user_prompt") -> None:
    s = user_prompt or ""
    cap = max_prompt_chars()
    if len(s) > cap:
        raise MCPInputError(
            "INPUT_TOO_LARGE",
            f"{field} exceeds limit ({len(s)} > {cap} chars). "
            f"Raise CONTEXTBUDDY_MCP_MAX_PROMPT_CHARS if needed.",
        )


def validate_context(context: Union[str, Sequence[str]], *, field: str = "context") -> None:
    total = context_total_chars(context)
    cap = max_context_total_chars()
    if total > cap:
        raise MCPInputError(
            "INPUT_TOO_LARGE",
            f"{field} exceeds limit ({total} > {cap} total chars). "
            f"Raise CONTEXTBUDDY_MCP_MAX_CONTEXT_CHARS if needed.",
        )


def validate_query_str(q: str, *, field: str = "query") -> None:
    s = q or ""
    cap = max_query_chars()
    if len(s) > cap:
        raise MCPInputError(
            "INPUT_TOO_LARGE",
            f"{field} exceeds limit ({len(s)} > {cap} chars). "
            f"Raise CONTEXTBUDDY_MCP_MAX_QUERY_CHARS if needed.",
        )


def run_with_timeout(seconds: float, fn: Callable[[], T]) -> T:
    """
    Run `fn` in a worker thread and enforce a wall-clock limit.

    Uses one thread per call (not a shared pool across requests). Embedders or
    SDK clients that are not thread-safe may misbehave under concurrent MCP
    calls; see docs/reference/mcp.md (MCP policy → timeouts).
    """
    if seconds <= 0:
        return fn()
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(fn)
        try:
            return fut.result(timeout=seconds)
        except FuturesTimeout as e:
            raise MCPToolTimeout(seconds) from e


def error_payload_compress(message: str, code: str = "ERROR") -> Dict[str, Any]:
    return {
        "ok": False,
        "error": {"code": code, "message": message},
        "prompt": "",
        "report": stub_report_dict(),
    }


def limits_summary() -> Dict[str, Any]:
    return {
        "max_prompt_chars": max_prompt_chars(),
        "max_context_total_chars": max_context_total_chars(),
        "max_query_chars": max_query_chars(),
        "tool_timeout_sec": tool_timeout_sec(),
        "index_timeout_sec": index_timeout_sec(),
    }
