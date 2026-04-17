from __future__ import annotations

from contextbuddy.benchmarks import BenchmarkCase, quality_gate, run_benchmarks
from contextbuddy.engine import ContextEngineConfig


def test_benchmarks_runner_and_gate_passes_default_case() -> None:
    cases = [
        BenchmarkCase(
            name="case",
            document="Invoice INV-92831 issued 2026-04-01 for account_id=acct_12345.\n\n"
            + ("noise " * 500),
            question="What is the invoice id?",
            expected_substring="INV-92831",
            required_entities=("INV-92831", "2026-04-01", "acct_12345"),
        )
    ]
    res = run_benchmarks(cases, config=ContextEngineConfig(max_context_tokens=200))
    ok, _ = quality_gate(res, min_answer_survival_rate=0.0, require_entity_survival_rate=100.0)
    assert ok


def test_quality_gate_fails_on_answer_rate() -> None:
    cases = [
        BenchmarkCase(
            name="miss",
            document="hello world",
            question="q",
            expected_substring="this substring will not exist",
        )
    ]
    res = run_benchmarks(cases, config=ContextEngineConfig(max_context_tokens=50))
    ok, msg = quality_gate(res, min_answer_survival_rate=100.0, require_entity_survival_rate=0.0)
    assert not ok
    assert "answer survival" in msg.lower()

