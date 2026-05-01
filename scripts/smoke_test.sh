#!/usr/bin/env bash
set -euo pipefail

MAX_TOKENS="${1:-200}"

echo "== ContextBuddy smoke test =="

python -m pip install -e . >/dev/null

CTX="Invoice INV-92831 issued 2026-04-01 for account_id=acct_12345.

$(python - <<'PY'
print("noise " * 8000)
PY
)"

echo "-- CLI compress (streaming)"
printf "%s" "$CTX" | python -m contextbuddy compress \
  --prompt "What is the invoice id and date?" \
  --max-tokens "$MAX_TOKENS" \
  --stream >/dev/null

echo "-- Bench gate"
python -m contextbuddy bench --gate >/dev/null

echo "OK"

