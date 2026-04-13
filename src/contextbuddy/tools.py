"""
Agent tool definitions compatible with OpenAI function calling.

Usage:
    tools = [make_search_tool(store), make_compress_tool(engine)]
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=tools,
    )
"""
from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional

from .engine import ContextEngine
from .store.memory import MemoryStore


def make_search_tool(
    store: MemoryStore,
    *,
    name: str = "search_documents",
    description: str = "Search the document store for relevant information.",
    top_k: int = 10,
) -> Dict[str, Any]:
    """
    Create an OpenAI-compatible tool definition for document search.

    Returns a dict with 'type', 'function' (schema), and '_callable' (the handler).
    """
    def _handler(query: str, **kwargs: Any) -> str:
        results = store.search(query, top_k=top_k)
        if not results:
            return "No relevant documents found."
        parts = []
        for i, r in enumerate(results, 1):
            source = r.metadata.get("source", "unknown")
            parts.append(f"[{i}] (score: {r.score:.2f}, source: {source})\n{r.chunk}")
        return "\n\n".join(parts)

    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to find relevant document chunks.",
                    },
                },
                "required": ["query"],
            },
        },
        "_callable": _handler,
    }


def make_compress_tool(
    engine: ContextEngine,
    *,
    name: str = "compress_context",
    description: str = "Compress a large context payload into a token-efficient version.",
) -> Dict[str, Any]:
    """
    Create an OpenAI-compatible tool definition for context compression.

    Returns a dict with 'type', 'function' (schema), and '_callable' (the handler).
    """
    def _handler(context: str, prompt: str = "Process the context.", **kwargs: Any) -> str:
        final_prompt, report = engine.build_prompt(
            user_prompt=prompt,
            context=context,
        )
        return json.dumps({
            "compressed_prompt": final_prompt,
            "original_tokens": report.original_prompt_tokens,
            "final_tokens": report.final_prompt_tokens,
            "reduction_pct": round(report.reduction_pct, 1),
            "entities_preserved": report.entities,
        })

    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    "context": {
                        "type": "string",
                        "description": "The raw context text to compress.",
                    },
                    "prompt": {
                        "type": "string",
                        "description": "The user prompt to optimize compression for.",
                    },
                },
                "required": ["context"],
            },
        },
        "_callable": _handler,
    }


def handle_tool_call(
    tool_call: Any,
    tools: List[Dict[str, Any]],
) -> str:
    """
    Dispatch an OpenAI tool_call to the matching _callable.

    Usage:
        for tc in response.choices[0].message.tool_calls:
            result = handle_tool_call(tc, tools)
    """
    fn_name = tool_call.function.name if hasattr(tool_call, "function") else tool_call.get("function", {}).get("name", "")
    args_str = tool_call.function.arguments if hasattr(tool_call, "function") else tool_call.get("function", {}).get("arguments", "{}")

    args = json.loads(args_str) if isinstance(args_str, str) else args_str

    for tool in tools:
        fn = tool.get("function", {})
        if fn.get("name") == fn_name:
            handler = tool.get("_callable")
            if handler:
                return handler(**args)

    return json.dumps({"error": f"Unknown tool: {fn_name}"})
