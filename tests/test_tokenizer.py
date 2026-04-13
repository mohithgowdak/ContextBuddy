import pytest

from contextbuddy.tokenizer import HeuristicTokenizer


def test_basic_count() -> None:
    t = HeuristicTokenizer(chars_per_token=4.0)
    assert t.count_tokens("hello world!!") == 3  # 13 chars / 4 = 3.25 -> 3


def test_minimum_one_token() -> None:
    t = HeuristicTokenizer()
    assert t.count_tokens("") >= 1


def test_custom_ratio() -> None:
    t = HeuristicTokenizer(chars_per_token=2.0)
    assert t.count_tokens("abcdef") == 3  # 6 / 2 = 3


def test_invalid_chars_per_token() -> None:
    t = HeuristicTokenizer(chars_per_token=0)
    with pytest.raises(ValueError):
        t.count_tokens("test")


def test_long_text() -> None:
    t = HeuristicTokenizer()
    text = "a" * 40000
    result = t.count_tokens(text)
    assert result == 10000  # 40000 / 4
