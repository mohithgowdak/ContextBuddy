from __future__ import annotations

import os
from pathlib import Path

import pytest

from contextbuddy.mcp.security import validate_root


def test_validate_root_dot_uses_single_allowed_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setenv("CONTEXTBUDDY_ALLOWED_ROOTS", str(repo))
    assert validate_root(".").resolve() == repo.resolve()


def test_validate_root_dot_ambiguous_with_multiple_allowed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    monkeypatch.setenv("CONTEXTBUDDY_ALLOWED_ROOTS", f"{a};{b}")
    with pytest.raises(PermissionError, match="ambiguous"):
        validate_root(".")


def test_validate_root_dot_without_allowlist_uses_cwd(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("CONTEXTBUDDY_ALLOWED_ROOTS", raising=False)
    sub = tmp_path / "sub"
    sub.mkdir()
    monkeypatch.chdir(sub)
    assert validate_root(".").resolve() == sub.resolve()
