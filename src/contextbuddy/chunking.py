from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Union


_para_split_re = re.compile(r"\n\s*\n+")


def _clean_chunk(s: str) -> str:
    return re.sub(r"[ \t]+\n", "\n", s.strip())


@dataclass(frozen=True)
class Chunker:
    min_chars: int = 40

    def chunk(self, context: Union[str, Sequence[str]]) -> List[str]:
        if isinstance(context, str):
            raw = [c for c in _para_split_re.split(context) if c.strip()]
        else:
            raw = list(context)

        out: List[str] = []
        for c in raw:
            cc = _clean_chunk(str(c))
            if len(cc) >= self.min_chars:
                out.append(cc)
        return out

