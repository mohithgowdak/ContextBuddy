"""
Example: async usage with arun() for non-blocking LLM calls.
"""
from __future__ import annotations

import asyncio
import os

from contextbuddy import ContextEngine, ContextEngineConfig


async def main() -> None:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])

    engine = ContextEngine(
        ContextEngineConfig(dev_mode=True, max_context_tokens=2000)
    )

    context = (
        "Invoice INV-92831 issued 2026-04-01 for account_id=acct_12345.\n\n"
        "Unrelated text about product roadmaps and Q3 planning.\n\n" * 30
        + "Ticket ACME-2041: chargebacks for user_id=usr_9z8y7x6w.\n\n"
    )

    response = await engine.arun(
        user_prompt="Summarize the invoice and ticket with all IDs.",
        context=context,
        llm_call=lambda prompt: client.responses.create(
            model="gpt-4o-mini",
            input=prompt,
        ),
    )
    print(response.output_text)


if __name__ == "__main__":
    asyncio.run(main())
