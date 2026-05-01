param(
  [switch]$BenchGate
)

$ErrorActionPreference = "Stop"

Write-Host "== ContextBuddy release check ==" -ForegroundColor Cyan

python -m pytest -q

if (Test-Path dist) { Remove-Item -Recurse -Force dist }
python -m build
python -m twine check dist/*

if ($BenchGate) {
  python -m contextbuddy bench --gate --json bench-report.json
}

Write-Host "OK" -ForegroundColor Green

