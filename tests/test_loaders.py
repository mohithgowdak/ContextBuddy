import json
import os
import tempfile
from pathlib import Path

import pytest

from contextbuddy.loaders import load
from contextbuddy.loaders.text import (
    load_text, load_csv, load_json, load_text_auto, _split_paragraphs,
)
from contextbuddy.loaders.directory import load_directory


def test_load_text_file() -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write("First paragraph.\n\nSecond paragraph.\n\nThird paragraph.")
        f.flush()
        path = f.name
    try:
        chunks = load_text(path)
        assert len(chunks) == 3
        assert chunks[0] == "First paragraph."
    finally:
        os.unlink(path)


def test_load_csv_file() -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8") as f:
        f.write("Name,Age,City\nAlice,30,NYC\nBob,25,LA\n")
        f.flush()
        path = f.name
    try:
        chunks = load_csv(path)
        assert len(chunks) == 2
        assert "Alice" in chunks[0]
        assert "Name: Alice" in chunks[0]
    finally:
        os.unlink(path)


def test_load_json_list() -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(["first item", "second item", "third item"], f)
        f.flush()
        path = f.name
    try:
        chunks = load_json(path)
        assert len(chunks) == 3
        assert chunks[0] == "first item"
    finally:
        os.unlink(path)


def test_load_json_dict() -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump({"title": "Test", "body": "Content here"}, f)
        f.flush()
        path = f.name
    try:
        chunks = load_json(path)
        assert any("title" in c for c in chunks)
    finally:
        os.unlink(path)


def test_load_auto_detects_txt() -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        f.write("Hello world paragraph one.\n\nParagraph two here.")
        f.flush()
        path = f.name
    try:
        chunks = load(path)
        assert len(chunks) >= 1
    finally:
        os.unlink(path)


def test_load_directory() -> None:
    with tempfile.TemporaryDirectory() as d:
        p1 = Path(d) / "a.txt"
        p1.write_text("Document A first paragraph.\n\nDocument A second paragraph.", encoding="utf-8")
        p2 = Path(d) / "b.txt"
        p2.write_text("Document B content here with details.", encoding="utf-8")

        chunks = load_directory(d)
        assert len(chunks) >= 2
        assert any("Document A" in c for c in chunks)
        assert any("Document B" in c for c in chunks)


def test_load_directory_prefixes_source() -> None:
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "notes.txt"
        p.write_text("Important notes about the project here.", encoding="utf-8")

        chunks = load_directory(d, prefix_source=True)
        assert any("[notes.txt]" in c for c in chunks)


def test_load_batch() -> None:
    with tempfile.TemporaryDirectory() as d:
        p1 = Path(d) / "a.txt"
        p1.write_text("Content from file A about machine learning.", encoding="utf-8")
        p2 = Path(d) / "b.txt"
        p2.write_text("Content from file B about data science topics.", encoding="utf-8")

        chunks = load([str(p1), str(p2)])
        assert len(chunks) >= 2


def test_load_nonexistent_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load("/nonexistent/file.txt")


def test_split_paragraphs() -> None:
    text = "A\n\nB\n\n\nC"
    parts = _split_paragraphs(text)
    assert parts == ["A", "B", "C"]
