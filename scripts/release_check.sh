#!/usr/bin/env bash
set -euo pipefail

BENCH_GATE="${1:-}"

echo "== ContextBuddy release check =="

pytest -q

rm -rf dist
python -m build
python -m twine check dist/*

if [[ "$BENCH_GATE" == "--bench" ]]; then
  python -m contextbuddy bench --gate --json bench-report.json
fi

echo "OK"

