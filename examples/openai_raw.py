from __future__ import annotations

import os

from contextbuddy import ContextEngine, ContextEngineConfig


def main() -> None:
    # pip install "contextbuddy[openai]" openai
    from openai import OpenAI

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    raw_context = """
    Invoice INV-92831 was issued on 2026-04-01 for account_id=acct_12345.

    Random blog post paragraph about unrelated topics...

    Support ticket ACME-2041 mentions repeated chargebacks for user_id=usr_9z8y7x6w.
    """

    prompt = "Summarize the invoice and the support ticket. Do not miss any IDs or dates."

    engine = ContextEngine(
        ContextEngineConfig(
            dev_mode=True,
            max_context_tokens=400,
            min_relevance=0.10,
        )
    )

    res = engine.run(
        user_prompt=prompt,
        context=raw_context,
        llm_call=lambda final_prompt: client.responses.create(
            model="gpt-4.1-mini",
            input=final_prompt,
        ),
    )

    print(res.output_text)


if __name__ == "__main__":
    main()

