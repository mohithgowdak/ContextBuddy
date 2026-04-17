from __future__ import annotations

from contextbuddy import ContextEngine, ContextEngineConfig


def test_build_prompt_stream_emits_done_with_prompt_and_report() -> None:
    engine = ContextEngine(ContextEngineConfig(max_context_tokens=200))
    context = ("Invoice INV-92831 issued 2026-04-01 for account_id=acct_12345.\n\n" + ("noise " * 500))

    events = list(engine.build_prompt_stream(user_prompt="What is the invoice id?", context=context))
    assert events[0].stage == "start"
    assert events[-1].stage == "done"
    assert events[-1].prompt and "User:" in events[-1].prompt
    assert events[-1].report is not None
    assert events[-1].report.final_prompt_tokens > 0

