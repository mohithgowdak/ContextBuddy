from contextbuddy.chunking import Chunker


def test_splits_paragraphs() -> None:
    text = "First paragraph here.\n\nSecond paragraph here.\n\nThird paragraph here."
    chunks = Chunker(min_chars=10).chunk(text)
    assert len(chunks) == 3
    assert chunks[0] == "First paragraph here."
    assert chunks[2] == "Third paragraph here."


def test_accepts_list() -> None:
    items = ["chunk one is long enough", "chunk two is long enough"]
    chunks = Chunker(min_chars=10).chunk(items)
    assert chunks == items


def test_filters_short_chunks() -> None:
    text = "ab\n\nThis is a proper paragraph.\n\nxy"
    chunks = Chunker(min_chars=20).chunk(text)
    assert len(chunks) == 1
    assert "proper paragraph" in chunks[0]


def test_strips_trailing_whitespace() -> None:
    text = "Hello world this is text.   \n\n   Another paragraph with content."
    chunks = Chunker(min_chars=10).chunk(text)
    for ch in chunks:
        assert ch == ch.strip()


def test_empty_input() -> None:
    assert Chunker().chunk("") == []
    assert Chunker().chunk([]) == []
