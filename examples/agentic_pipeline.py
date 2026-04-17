"""
End-to-end example: an agentic pipeline wrapper.

Scenario
--------
A tiny agent that can answer questions over a bundle of "company docs"
by (1) retrieving candidate chunks from a vector store,
(2) compressing them with ContextBuddy (entity-safe, budgeted),
(3) calling an LLM (stubbed here so this runs with zero credentials),
(4) looping if the LLM says "need more context".

The point of this example is the *wrapper pattern*: wrap every LLM call
in the agent with ContextBuddy compression so you keep token cost low
while never dropping the one ID / date / amount that actually matters.

Run
---
    python examples/agentic_pipeline.py

Swap in a real LLM
------------------
Replace `stub_llm` with any callable `str -> str`. For example:

    from openai import OpenAI
    client = OpenAI()
    def real_llm(prompt: str) -> str:
        r = client.responses.create(model="gpt-4o-mini", input=prompt)
        return r.output_text

    agent = ContextAgent(store, llm=real_llm, dev_mode=True)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

from contextbuddy import (
    ContextEngine,
    ContextEngineConfig,
    MemoryStore,
)


# -----------------------------------------------------------------------------
# 1. Sample "company docs" — the kind of messy context a real agent sees.
# -----------------------------------------------------------------------------
COMPANY_DOCS: List[str] = [
    "Invoice INV-92831 issued 2026-04-01 for account_id=acct_12345. "
    "Amount: $4,500.00 USD. Payment due within 30 days.",

    "Support ticket ACME-2041: customer reported repeated chargebacks "
    "for user_id=usr_9z8y7x6w. Escalation required.",

    "Quarterly planning notes: the team discussed OKRs for Q3, hiring "
    "plans, and office renovation timelines. No decisions yet.",

    "Team lunch menu for next week: Monday pizza, Tuesday sushi, "
    "Wednesday salads, Thursday burritos, Friday open.",

    "Incident IR-778: brief API outage at 2026-04-08T14:20 lasting 6 minutes. "
    "Root cause: misconfigured rate limiter. Fix deployed.",

    "Random newsletter content about industry trends, conferences, and "
    "general thought-leadership. Mostly filler.",
]


# -----------------------------------------------------------------------------
# 2. A stub LLM so this example runs without any API key.
#    It returns deterministic strings that trigger the agent loop.
# -----------------------------------------------------------------------------
def stub_llm(prompt: str) -> str:
    """Pretend LLM: reads the compressed prompt and writes a structured answer."""
    low = prompt.lower()

    # If the compressed prompt mentions an invoice and a ticket, answer directly.
    if "inv-92831" in low and "acme-2041" in low:
        return (
            "ANSWER: Invoice INV-92831 (2026-04-01, $4,500.00, acct_12345) and "
            "Ticket ACME-2041 (chargebacks, usr_9z8y7x6w) are both active."
        )

    # Otherwise, ask the agent for more context.
    return "NEED_MORE_CONTEXT: I can't see invoice + ticket IDs yet."


# -----------------------------------------------------------------------------
# 3. The agent: every LLM call is wrapped by ContextBuddy compression.
# -----------------------------------------------------------------------------
@dataclass
class AgentStep:
    iteration: int
    retrieved_chunks: int
    tokens_before: int
    tokens_after: int
    reduction_pct: float
    estimated_savings: float
    llm_output: str


class ContextAgent:
    """A minimal ReAct-style loop with ContextBuddy compression on every step."""

    def __init__(
        self,
        store: MemoryStore,
        *,
        llm: Callable[[str], str],
        max_iterations: int = 3,
        max_context_tokens: int = 800,
        dev_mode: bool = False,
    ) -> None:
        self.store = store
        self.llm = llm
        self.max_iterations = max_iterations
        self.engine = ContextEngine(
            ContextEngineConfig(
                max_context_tokens=max_context_tokens,
                dev_mode=dev_mode,
            )
        )
        self.steps: List[AgentStep] = []

    def run(self, question: str) -> str:
        top_k = 3
        answer: Optional[str] = None

        for i in range(1, self.max_iterations + 1):
            results = self.store.search(question, top_k=top_k)
            chunks = [r.chunk for r in results]

            final_prompt, report = self.engine.build_prompt(
                user_prompt=question,
                context=chunks,
            )

            output = self.llm(final_prompt)

            self.steps.append(
                AgentStep(
                    iteration=i,
                    retrieved_chunks=len(chunks),
                    tokens_before=report.original_prompt_tokens,
                    tokens_after=report.final_prompt_tokens,
                    reduction_pct=report.reduction_pct,
                    estimated_savings=report.estimated_savings,
                    llm_output=output,
                )
            )

            if not output.startswith("NEED_MORE_CONTEXT"):
                answer = output
                break

            top_k = min(top_k * 2, 20)

        return answer or "AGENT_FAILED: exceeded max iterations."


# -----------------------------------------------------------------------------
# 4. Wire it up and print a mini report per step.
# -----------------------------------------------------------------------------
def main() -> None:
    store = MemoryStore()
    store.add(COMPANY_DOCS, metadata={"source": "company_docs"})

    agent = ContextAgent(store, llm=stub_llm, dev_mode=False, max_context_tokens=800)
    answer = agent.run("Summarize the invoice and support ticket with IDs.")

    print("=" * 64)
    print("Per-step compression report")
    print("=" * 64)
    for s in agent.steps:
        print(
            f"  step {s.iteration}: retrieved={s.retrieved_chunks} "
            f"tokens {s.tokens_before} -> {s.tokens_after} "
            f"({s.reduction_pct:.1f}% smaller, ~${s.estimated_savings:.4f} saved)"
        )
    print("-" * 64)
    print("Final answer:")
    print(answer)


if __name__ == "__main__":
    main()
