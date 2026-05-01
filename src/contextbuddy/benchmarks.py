from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .engine import ContextEngine, ContextEngineConfig


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    document: str
    question: str
    expected_substring: str
    required_entities: Sequence[str] = ()


@dataclass(frozen=True)
class BenchmarkResult:
    cases: int
    answer_survival_rate: float
    entity_survival_rate: float
    mean_prompt_reduction_pct: float
    mean_latency_ms: float
    p95_latency_ms: float
    failures: List[Dict[str, Any]]


def default_dataset() -> List[BenchmarkCase]:
    """
    Small, zero-dep benchmark suite.

    This is intentionally simple: it measures whether the compressed prompt
    still contains the answer-bearing text (substring proxy), and whether
    injected entities survive.
    """
    # Make docs long enough that the default bench budget forces pruning,
    # otherwise mean_reduction becomes ~0% and the harness is misleading.
    noise = "\n\n".join([("Unrelated boilerplate text. " * 80) for _ in range(25)])
    code_noise = "\n".join(["# filler comment"] * 1200)

    return [
        BenchmarkCase(
            name="legal/payment-terms",
            document=(
                "ARTICLE II - PAYMENT TERMS\n"
                "2.1 Payment due within 30 days of invoice date.\n"
                "2.2 Late fee applies after 45 days.\n\n"
                "ARTICLE III - TERMINATION\n"
                "3.1 Either Party may terminate with 30 days notice.\n"
            )
            + ("\n\n" + noise),
            question="What are the payment terms?",
            expected_substring="Payment due within 30 days",
        ),
        BenchmarkCase(
            name="support/invoice-id",
            document=(
                "Ticket ACME-2041: chargebacks for user_id=usr_9z8y7x6w.\n\n"
                "Invoice INV-92831 issued 2026-04-01 for account_id=acct_12345. "
                "Amount: $4,500.00 USD. Payment due within 30 days.\n\n"
            )
            + ("\n\n" + noise),
            question="What is the invoice ID and date?",
            expected_substring="INV-92831 issued 2026-04-01",
            required_entities=("INV-92831", "2026-04-01", "acct_12345"),
        ),
        BenchmarkCase(
            name="python/code-def",
            document=(
                "import os\n\n"
                "def compute_total(x: int) -> int:\n"
                "    y = x + 1\n"
                "    return y\n\n"
                "def other() -> str:\n"
                "    return 'ok'\n\n"
            )
            + ("\n" + code_noise + "\n"),
            question="What does compute_total return?",
            expected_substring="return y",
        ),
    ]


def load_dataset(path: str) -> List[BenchmarkCase]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.loads(f.read())
    out: List[BenchmarkCase] = []
    for item in raw:
        out.append(
            BenchmarkCase(
                name=str(item["name"]),
                document=str(item["document"]),
                question=str(item["question"]),
                expected_substring=str(item["expected_substring"]),
                required_entities=tuple(item.get("required_entities", ())),
            )
        )
    return out


def _percent(num: int, den: int) -> float:
    return 0.0 if den <= 0 else (num / den) * 100.0


def _p95(values: List[float]) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    idx = int(0.95 * (len(xs) - 1))
    return xs[idx]


def run_benchmarks(
    cases: Sequence[BenchmarkCase],
    *,
    config: Optional[ContextEngineConfig] = None,
) -> BenchmarkResult:
    # Bench harness runs in conservative mode by default to minimize silent recall misses.
    engine = ContextEngine(
        config
        or ContextEngineConfig(
            max_context_tokens=1200,
            dev_mode=False,
            conservative_mode=True,
        )
    )

    answer_ok = 0
    entity_ok = 0
    total_entities = 0
    reductions: List[float] = []
    latencies_ms: List[float] = []
    failures: List[Dict[str, Any]] = []

    for c in cases:
        t0 = time.perf_counter()
        prompt, report = engine.build_prompt(user_prompt=c.question, context=c.document)
        dt_ms = (time.perf_counter() - t0) * 1000.0

        latencies_ms.append(dt_ms)
        reductions.append(float(report.reduction_pct))

        ans_survives = c.expected_substring in prompt
        if ans_survives:
            answer_ok += 1
        else:
            failures.append(
                {
                    "case": c.name,
                    "type": "ANSWER_MISS",
                    "expected_substring": c.expected_substring,
                }
            )

        for ent in c.required_entities:
            total_entities += 1
            if ent in prompt and ent in report.entities:
                entity_ok += 1
            else:
                failures.append(
                    {
                        "case": c.name,
                        "type": "ENTITY_MISS",
                        "entity": ent,
                    }
                )

    return BenchmarkResult(
        cases=len(cases),
        answer_survival_rate=_percent(answer_ok, len(cases)),
        entity_survival_rate=_percent(entity_ok, total_entities) if total_entities else 100.0,
        mean_prompt_reduction_pct=(sum(reductions) / len(reductions)) if reductions else 0.0,
        mean_latency_ms=(sum(latencies_ms) / len(latencies_ms)) if latencies_ms else 0.0,
        p95_latency_ms=_p95(latencies_ms),
        failures=failures,
    )


def quality_gate(
    result: BenchmarkResult,
    *,
    min_answer_survival_rate: float = 85.0,
    require_entity_survival_rate: float = 100.0,
) -> Tuple[bool, str]:
    """
    Returns (ok, message). Intended for release gating.
    """
    if result.entity_survival_rate + 1e-9 < require_entity_survival_rate:
        return (
            False,
            f"FAIL: entity survival {result.entity_survival_rate:.1f}% < {require_entity_survival_rate:.1f}%",
        )
    if result.answer_survival_rate + 1e-9 < min_answer_survival_rate:
        return (
            False,
            f"FAIL: answer survival {result.answer_survival_rate:.1f}% < {min_answer_survival_rate:.1f}%",
        )
    return True, "PASS"


def format_summary(result: BenchmarkResult) -> str:
    return (
        f"cases={result.cases} | "
        f"answer_survival={result.answer_survival_rate:.1f}% | "
        f"entity_survival={result.entity_survival_rate:.1f}% | "
        f"mean_reduction={result.mean_prompt_reduction_pct:.1f}% | "
        f"latency_mean={result.mean_latency_ms:.1f}ms p95={result.p95_latency_ms:.1f}ms"
    )

