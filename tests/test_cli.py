import os
import tempfile

from contextbuddy.cli import main


def test_compress_with_inline_context(capsys) -> None:
    os.environ["NO_COLOR"] = "1"
    try:
        code = main([
            "compress",
            "--prompt", "What is the invoice number?",
            "--context", (
                "Invoice INV-92831 was issued on 2026-04-01 for account_id=acct_12345.\n\n"
                "Some completely unrelated filler text about weather and sports.\n\n"
                "Support ticket ACME-2041 mentions repeated chargebacks."
            ),
            "--max-tokens", "200",
            "--no-color",
        ])
        assert code == 0
    finally:
        os.environ.pop("NO_COLOR", None)


def test_compress_with_file(capsys) -> None:
    os.environ["NO_COLOR"] = "1"
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write("Paragraph one about machine learning and data science.\n\n")
            f.write("Paragraph two about cooking recipes and kitchen equipment.\n\n")
            f.write("Paragraph three about machine learning model training.\n\n")
            f.flush()
            path = f.name

        code = main([
            "compress",
            "--prompt", "What about machine learning?",
            "--file", path,
            "--show-prompt",
            "--no-color",
        ])
        assert code == 0
        captured = capsys.readouterr()
        assert "User:" in captured.out
    finally:
        os.environ.pop("NO_COLOR", None)
        try:
            os.unlink(path)
        except OSError:
            pass


def test_compress_missing_file() -> None:
    code = main(["compress", "--prompt", "test", "--file", "/nonexistent/path.txt"])
    assert code == 1


def test_no_command_shows_help(capsys) -> None:
    code = main([])
    assert code == 0
