"""
Convenience wrappers that transparently compress context before hitting the LLM.

Usage:
    from contextbuddy.wrappers import wrap_openai

    client = wrap_openai(OpenAI(), max_context_tokens=4000, dev_mode=True)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": huge_context},
            {"role": "user", "content": "Summarize the key points."},
        ],
    )
    # ContextBuddy automatically compressed the system message.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from .engine import ContextEngine, ContextEngineConfig
from .pricing import get_pricing
from .types import Embedder, ModelPricing, Tokenizer


class _WrappedCompletions:
    """Proxy for client.chat.completions that compresses system/context messages."""

    def __init__(self, original_completions: Any, engine: ContextEngine):
        self._orig = original_completions
        self._engine = engine

    def create(self, *, messages: List[Dict[str, Any]], **kwargs: Any) -> Any:
        compressed_messages = self._compress_messages(messages)
        return self._orig.create(messages=compressed_messages, **kwargs)

    async def acreate(self, *, messages: List[Dict[str, Any]], **kwargs: Any) -> Any:
        compressed_messages = self._compress_messages(messages)
        create_fn = getattr(self._orig, "acreate", None) or self._orig.create
        return await create_fn(messages=compressed_messages, **kwargs)

    def _compress_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        system_parts: List[str] = []
        user_prompt = ""
        other_messages: List[Dict[str, Any]] = []

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "system":
                system_parts.append(str(content))
            elif role == "user" and not user_prompt:
                user_prompt = str(content)
                other_messages.append(msg)
            else:
                other_messages.append(msg)

        if not system_parts:
            return list(messages)

        context = "\n\n".join(system_parts)
        if not user_prompt:
            user_prompt = "Process the context."

        final_prompt, report = self._engine.build_prompt(
            user_prompt=user_prompt,
            context=context,
        )
        self._engine.last_report = report
        self._engine._emit_report(report)

        result = [{"role": "system", "content": final_prompt}]
        result.extend(other_messages)
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self._orig, name)


class _WrappedChat:
    """Proxy for client.chat."""

    def __init__(self, original_chat: Any, engine: ContextEngine):
        self.completions = _WrappedCompletions(original_chat.completions, engine)
        self._orig = original_chat

    def __getattr__(self, name: str) -> Any:
        return getattr(self._orig, name)


class WrappedClient:
    """Proxy for an OpenAI-style client with transparent context compression."""

    def __init__(self, client: Any, engine: ContextEngine):
        self._client = client
        self.engine = engine
        self.chat = _WrappedChat(client.chat, engine)

    @property
    def last_report(self):
        return self.engine.last_report

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


def wrap_openai(
    client: Any,
    *,
    max_context_tokens: int = 4000,
    min_relevance: float = 0.15,
    dev_mode: bool = True,
    pricing: Optional[ModelPricing] = None,
    embedder: Optional[Embedder] = None,
    tokenizer: Optional[Tokenizer] = None,
) -> WrappedClient:
    """
    Wrap an OpenAI client so chat completions automatically compress
    system-message context before sending.

    Returns a WrappedClient that behaves like the original client.
    """
    config = ContextEngineConfig(
        max_context_tokens=max_context_tokens,
        min_relevance=min_relevance,
        dev_mode=dev_mode,
        pricing=pricing or get_pricing("gpt-4o-mini"),
    )
    engine = ContextEngine(config, embedder=embedder, tokenizer=tokenizer)
    return WrappedClient(client, engine)
