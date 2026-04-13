#!/usr/bin/env bash
# Quick CLI demo — run from the repo root.
# No API keys needed. Shows compression + ROI output.

echo "Invoice INV-92831 issued on 2026-04-01 for account_id=acct_12345.
Amount: \$4,500.00 USD. Payment due within 30 days.

Some irrelevant text about the weather forecast for next week.
Rain is expected on Tuesday and Wednesday. Nothing to do with invoices.

More filler paragraphs about office renovation and lunch menus.
The cafeteria will be closed for remodeling from April 15 to April 30.

Support ticket ACME-2041: customer reported repeated chargebacks
for user_id=usr_9z8y7x6w. Escalation required immediately.

Additional filler about company picnic planning and team building events.
The annual picnic will be held on May 15th at the park." \
  | python -m contextbuddy compress \
      --prompt "Summarize the invoice and the support ticket. Keep all IDs." \
      --max-tokens 200 \
      --show-prompt
