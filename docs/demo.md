# Demo (GIF) — README-ready

This project looks 10× more credible with a short terminal GIF showing:
- smooth progress bar (`--stream`)
- ROI box output
- (optional) the compressed prompt

## What to record (recommended)

Command:

```bash
python -m pip install -e .

echo "Invoice INV-92831 issued 2026-04-01 for account_id=acct_12345.

Ticket ACME-2041: chargebacks for user_id=usr_9z8y7x6w.

"$(python - <<'PY'
print("noise " * 8000)
PY
)" | python -m contextbuddy compress \
  --prompt "Summarize invoice + ticket. Keep all IDs/dates." \
  --max-tokens 200 \
  --stream
```

Expected:
- one-line animated bar while compressing
- ROI report box (tokens before/after, reduction %, savings)

## Record on Windows (fastest)

1. Open Windows Terminal (PowerShell).
2. Start recording with **Win+Alt+R** (Xbox Game Bar).
3. Run the command above.
4. Stop recording (Win+Alt+R).
5. Convert MP4 → GIF with any converter (online or local).

## Where to put the file

Save as:
- `assets/cli-demo.gif`

Then the README will pick it up automatically (it links to this path).

