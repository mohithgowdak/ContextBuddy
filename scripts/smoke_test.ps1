param(
  [int]$MaxTokens = 200
)

$ErrorActionPreference = "Stop"

Write-Host "== ContextBuddy smoke test ==" -ForegroundColor Cyan

python -m pip install -e . > $null

$ctx = "Invoice INV-92831 issued 2026-04-01 for account_id=acct_12345.`n`n" + ("noise " * 8000)

Write-Host "-- CLI compress (streaming)" -ForegroundColor Cyan
$ctx | python -m contextbuddy compress --prompt "What is the invoice id and date?" --max-tokens $MaxTokens --stream | Out-Null

Write-Host "-- Bench gate" -ForegroundColor Cyan
python -m contextbuddy bench --gate | Out-Null

Write-Host "OK" -ForegroundColor Green

