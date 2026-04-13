from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from .engine import ContextReport


_SUPPORTS_COLOR: Optional[bool] = None


def _color_supported() -> bool:
    global _SUPPORTS_COLOR
    if _SUPPORTS_COLOR is not None:
        return _SUPPORTS_COLOR
    try:
        import os

        if os.environ.get("NO_COLOR"):
            _SUPPORTS_COLOR = False
            return False
        if os.environ.get("FORCE_COLOR"):
            _SUPPORTS_COLOR = True
            return True
        _SUPPORTS_COLOR = hasattr(sys.stderr, "isatty") and sys.stderr.isatty()
    except Exception:
        _SUPPORTS_COLOR = False
    return _SUPPORTS_COLOR


class _C:
    """ANSI color helpers."""

    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    CYAN = "\033[36m"
    WHITE = "\033[97m"
    RED = "\033[31m"
    MAGENTA = "\033[35m"
    BG_BLACK = "\033[40m"


def _c(code: str, text: str) -> str:
    if not _color_supported():
        return text
    return f"{code}{text}{_C.RESET}"


def _bar(filled: int, total: int, width: int = 20) -> str:
    if total <= 0:
        return " " * width
    ratio = max(0.0, min(1.0, filled / total))
    n = int(ratio * width)
    bar = "\u2588" * n + "\u2591" * (width - n)
    if _color_supported():
        return f"{_C.GREEN}{bar}{_C.RESET}"
    return bar


def format_report(report: ContextReport) -> str:
    lines: List[str] = []

    title = " ContextBuddy "
    top = f"\u250c{'─' * 58}\u2510"
    bot = f"\u2514{'─' * 58}\u2518"
    sep = f"\u251c{'─' * 58}\u2524"
    pad = "\u2502"

    lines.append(_c(_C.CYAN + _C.BOLD, top))
    lines.append(
        _c(_C.CYAN, pad)
        + _c(_C.BOLD + _C.WHITE, f"  {title:^54}  ")
        + _c(_C.CYAN, pad)
    )
    lines.append(_c(_C.CYAN, sep))

    before = report.original_prompt_tokens
    after = report.final_prompt_tokens
    pct = report.reduction_pct

    lines.append(
        _c(_C.CYAN, pad)
        + f"  Tokens   {_c(_C.DIM, 'before')} {_c(_C.YELLOW, str(before)):>8s}"
        + f"   {_c(_C.DIM, 'after')} {_c(_C.GREEN + _C.BOLD, str(after)):>8s}"
        + " " * (58 - 46)
        + _c(_C.CYAN, pad)
    )

    reduction_str = f"-{pct:.1f}%"
    savings_str = f"${report.estimated_savings:.4f}"
    lines.append(
        _c(_C.CYAN, pad)
        + f"  Saved    {_c(_C.GREEN + _C.BOLD, reduction_str):<22s}"
        + f"  Est. {_c(_C.GREEN, savings_str):<20s}"
        + _c(_C.CYAN, pad)
    )

    bar_vis = _bar(before - after, before, width=30)
    lines.append(
        _c(_C.CYAN, pad)
        + f"  [{bar_vis}] "
        + _c(_C.DIM, f"{before - after} tokens freed")
        + " " * max(0, 58 - 38 - len(str(before - after)) - len(" tokens freed"))
        + _c(_C.CYAN, pad)
    )

    lines.append(_c(_C.CYAN, sep))

    lines.append(
        _c(_C.CYAN, pad)
        + f"  Chunks   {_c(_C.DIM, 'total')} {report.total_chunks:<5d}"
        + f"  {_c(_C.DIM, 'kept')} {_c(_C.WHITE, str(report.kept_chunks)):<5s}"
        + f"  {_c(_C.DIM, 'pruned')} {_c(_C.RED, str(report.total_chunks - report.kept_chunks)):<5s}"
        + " " * 5
        + _c(_C.CYAN, pad)
    )

    if report.entities:
        ent_str = ", ".join(report.entities[:5])
        if len(report.entities) > 5:
            ent_str += f" +{len(report.entities) - 5} more"
        lines.append(
            _c(_C.CYAN, pad)
            + f"  Entities {_c(_C.MAGENTA, ent_str):<48s}"
            + _c(_C.CYAN, pad)
        )

    lines.append(_c(_C.CYAN + _C.BOLD, bot))

    return "\n".join(lines)


def print_report(report: ContextReport) -> None:
    sys.stderr.write(format_report(report) + "\n")
