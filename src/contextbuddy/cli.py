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
import time
import threading
import queue
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
    compress.add_argument("--stream", action="store_true", help="Show a live progress bar while compressing.")

    bench = sub.add_parser("bench", help="Run quality benchmarks and print a summary.")
    bench.add_argument("--dataset", type=str, default=None, help="Path to a JSON dataset file (optional).")
    bench.add_argument("--max-tokens", "-t", type=int, default=1200, help="Max context tokens (default 1200).")
    bench.add_argument("--min-answer", type=float, default=85.0, help="Min answer survival rate (default 85).")
    bench.add_argument("--require-entity", type=float, default=100.0, help="Required entity survival rate (default 100).")
    bench.add_argument("--gate", action="store_true", help="Exit non-zero if thresholds are not met.")
    bench.add_argument("--json", type=str, default=None, help="Write JSON report to this path.")

    return root


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 0

    if args.command == "compress":
        return _cmd_compress(args)
    if args.command == "bench":
        return _cmd_bench(args)

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
    if args.stream:
        final_prompt, report = _compress_with_progress(engine, args.prompt, context)
    else:
        final_prompt, report = engine.build_prompt(user_prompt=args.prompt, context=context)

    sys.stderr.write(format_report(report) + "\n")

    if args.show_prompt:
        sys.stdout.write(final_prompt)

    return 0


def _compress_with_progress(engine, user_prompt: str, context: str):
    """
    Render a live progress bar for compression.

    This is UX sugar for demos/playgrounds. It does not change output.
    """
    # If we don't have a TTY, don't attempt animation.
    if not (hasattr(sys.stderr, "isatty") and sys.stderr.isatty()):
        final_prompt, report = engine.build_prompt(user_prompt=user_prompt, context=context)
        return final_prompt, report

    stages = ["start", "chunked", "scored", "selected", "done"]
    stage_to_i = {s: i for i, s in enumerate(stages)}
    spinner = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def bar(i: int, total: int, width: int = 28) -> str:
        if total <= 0:
            return "[" + (" " * width) + "]"
        ratio = max(0.0, min(1.0, i / total))
        filled = int(width * ratio)
        filled = max(0, min(width, filled))
        return "[" + ("█" * filled) + ("░" * (width - filled)) + "]"

    q: "queue.Queue[object]" = queue.Queue()

    def worker() -> None:
        try:
            for ev in engine.build_prompt_stream(user_prompt=user_prompt, context=context):
                q.put(ev)
        except BaseException as e:  # pragma: no cover
            q.put(e)
        finally:
            q.put(None)  # sentinel

    t = threading.Thread(target=worker, daemon=True)
    t.start()

    last_line_len = 0
    final_prompt = ""
    report = None
    cur_stage = "start"
    cur_msg = "Starting compression"
    cur_stats = ""
    spin_i = 0
    last_draw = 0.0

    while True:
        try:
            item = q.get(timeout=0.05)
        except queue.Empty:
            item = "__tick__"

        if item is None:
            break
        if isinstance(item, BaseException):  # pragma: no cover
            raise item
        if item != "__tick__":
            ev = item
            # duck-typed access; ev is CompressionEvent
            cur_stage = getattr(ev, "stage", cur_stage)
            cur_msg = getattr(ev, "message", cur_msg) or cur_stage
            ev_report = getattr(ev, "report", None)
            ev_prompt = getattr(ev, "prompt", None)
            if ev_prompt is not None:
                final_prompt = ev_prompt
            if ev_report is not None:
                report = ev_report
                cur_stats = (
                    f"{report.original_prompt_tokens}->{report.final_prompt_tokens} "
                    f"({report.reduction_pct:.1f}%)"
                )

        now = time.perf_counter()
        # Throttle redraw to keep it smooth but cheap.
        if now - last_draw >= 0.03:
            i = stage_to_i.get(cur_stage, 0)
            b = bar(i, len(stages) - 1)
            sp = spinner[spin_i % len(spinner)]
            spin_i += 1

            stats_part = (cur_stats + " ") if cur_stats else ""
            line = f"{sp} {b} {cur_stage:<8s} {stats_part}{cur_msg}"

            pad = max(0, last_line_len - len(line))
            sys.stderr.write("\r" + line + (" " * pad))
            sys.stderr.flush()
            last_line_len = len(line)
            last_draw = now

    sys.stderr.write("\n")
    if report is None:
        final_prompt, report = engine.build_prompt(user_prompt=user_prompt, context=context)
    return final_prompt, report


def _cmd_bench(args: argparse.Namespace) -> int:
    from .benchmarks import (
        default_dataset,
        format_summary,
        load_dataset,
        quality_gate,
        run_benchmarks,
    )
    from .engine import ContextEngineConfig

    cases = load_dataset(args.dataset) if args.dataset else default_dataset()
    cfg = ContextEngineConfig(max_context_tokens=int(args.max_tokens), dev_mode=False)
    result = run_benchmarks(cases, config=cfg)

    sys.stdout.write(format_summary(result) + "\n")

    if args.json:
        Path(args.json).write_text(
            __import__("json").dumps(result.__dict__, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    if args.gate:
        ok, msg = quality_gate(
            result,
            min_answer_survival_rate=float(args.min_answer),
            require_entity_survival_rate=float(args.require_entity),
        )
        if not ok:
            sys.stderr.write(msg + "\n")
            return 2
        sys.stderr.write(msg + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
