from __future__ import annotations

from dataclasses import dataclass

from .types import Tokenizer


@dataclass(frozen=True)
class HeuristicTokenizer:
    """
    Fast, dependency-free token estimate.

    Rule of thumb: 1 token ~= 4 characters in English text.
    """

    chars_per_token: float = 4.0

    def count_tokens(self, text: str) -> int:
        cpt = float(self.chars_per_token)
        if cpt <= 0:
            raise ValueError("chars_per_token must be > 0")
        return max(1, int(len(text) / cpt))


class TiktokenTokenizer:
    """
    Optional accurate tokenizer via `tiktoken`.

    Install: `pip install contextbuddy[tiktoken]`
    """

    def __init__(self, *, encoding_name: str = "cl100k_base"):
        try:
            import tiktoken  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "TiktokenTokenizer requires the 'tiktoken' package. "
                "Install with: pip install contextbuddy[tiktoken]"
            ) from e

        self._enc = tiktoken.get_encoding(encoding_name)

    def count_tokens(self, text: str) -> int:
        return len(self._enc.encode(text))
