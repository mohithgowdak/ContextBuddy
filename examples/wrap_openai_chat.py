"""
Example: wrap_openai() — transparent context compression for chat completions.

The wrapped client compresses system messages automatically.
You don't change your existing code at all.
"""
from __future__ import annotations

import os

from openai import OpenAI

from contextbuddy import wrap_openai

client = wrap_openai(
    OpenAI(api_key=os.environ["OPENAI_API_KEY"]),
    max_context_tokens=2000,
    dev_mode=True,
)

huge_context = (
    "Invoice INV-92831 issued on 2026-04-01 for account_id=acct_12345. "
    "Amount: $4,500.00 USD. Payment due within 30 days.\n\n"
    "Some irrelevant text about the weather forecast for next week. "
    "Rain is expected on Tuesday and Wednesday.\n\n" * 20
    + "Support ticket ACME-2041: customer reported repeated chargebacks "
    "for user_id=usr_9z8y7x6w. Escalation required.\n\n"
    "More filler text about office renovation plans and lunch menus. " * 15
)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": huge_context},
        {"role": "user", "content": "Summarize the invoice and support ticket. Include all IDs and dates."},
    ],
)

print(response.choices[0].message.content)
