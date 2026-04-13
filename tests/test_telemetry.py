import os

from contextbuddy.engine import ContextReport
from contextbuddy.telemetry import format_report


def _sample_report() -> ContextReport:
    return ContextReport(
        original_prompt_tokens=15000,
        final_prompt_tokens=3000,
        original_context_tokens=14000,
        final_context_tokens=2000,
        reduction_pct=80.0,
        estimated_savings=0.06,
        kept_chunks=4,
        total_chunks=12,
        entities=["INV-92831", "2026-04-01", "acct_12345"],
        selected_indices=[0, 2, 5, 7],
    )


def test_format_report_contains_key_info() -> None:
    os.environ["NO_COLOR"] = "1"
    try:
        output = format_report(_sample_report())
        assert "15000" in output
        assert "3000" in output
        assert "80.0" in output
        assert "INV-92831" in output
        assert "ContextBuddy" in output
    finally:
        os.environ.pop("NO_COLOR", None)


def test_format_report_contains_box_drawing() -> None:
    os.environ["NO_COLOR"] = "1"
    try:
        output = format_report(_sample_report())
        assert "\u250c" in output  # top-left corner
        assert "\u2514" in output  # bottom-left corner
    finally:
        os.environ.pop("NO_COLOR", None)
