"""
contextbuddy CLI -- instant demo tool.

Usage:
    python -m contextbuddy compress --prompt "..." --file context.txt
    python -m contextbuddy compress --prompt "..." --context "inline text"
    python -m contextbuddy compress --prompt "..." --file context.txt --max-tokens 2000 --model gpt-4o
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="contextbuddy",
        description="Context compression & routing middleware for LLM calls.",
    )
    sub = root.add_subparsers(dest="command")

    compress = sub.add_parser("compress", help="Compress context and print the result.")
    compress.add_argument("--prompt", "-p", required=True, help="User prompt / question.")
    compress.add_argument("--file", "-f", type=str, default=None, help="Path to a text file with context.")
    compress.add_argument("--context", "-c", type=str, default=None, help="Inline context string.")
    compress.add_argument("--max-tokens", "-t", type=int, default=4000, help="Max context tokens (default 4000).")
    compress.add_argument("--min-relevance", type=float, default=0.15, help="Min relevance threshold (default 0.15).")
    compress.add_argument("--model", "-m", type=str, default="gpt-4o-mini", help="Model name for pricing (default gpt-4o-mini).")
    compress.add_argument("--no-color", action="store_true", help="Disable colored output.")
    compress.add_argument("--show-prompt", action="store_true", help="Print the compressed prompt to stdout.")

    return root


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 0

    if args.command == "compress":
        return _cmd_compress(args)

    parser.print_help()
    return 1


def _cmd_compress(args: argparse.Namespace) -> int:
    import os

    if args.no_color:
        os.environ["NO_COLOR"] = "1"

    from .engine import ContextEngine, ContextEngineConfig
    from .pricing import get_pricing
    from .telemetry import format_report

    context: str | None = None
    if args.file:
        p = Path(args.file)
        if not p.exists():
            sys.stderr.write(f"Error: file not found: {args.file}\n")
            return 1
        context = p.read_text(encoding="utf-8", errors="replace")
    elif args.context:
        context = args.context
    else:
        if not sys.stdin.isatty():
            context = sys.stdin.read()
        else:
            sys.stderr.write("Error: provide --file, --context, or pipe stdin.\n")
            return 1

    pricing = get_pricing(args.model)
    config = ContextEngineConfig(
        max_context_tokens=args.max_tokens,
        min_relevance=args.min_relevance,
        dev_mode=False,
        pricing=pricing,
    )
    engine = ContextEngine(config)
    final_prompt, report = engine.build_prompt(user_prompt=args.prompt, context=context)

    sys.stderr.write(format_report(report) + "\n")

    if args.show_prompt:
        sys.stdout.write(final_prompt)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
