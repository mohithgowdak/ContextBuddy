from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Union, Optional


_para_split_re = re.compile(r"\n\s*\n+")


def _clean_chunk(s: str) -> str:
    return re.sub(r"[ \t]+\n", "\n", s.strip())

def _merge_small_chunks(chunks: List[str], *, min_merge_chars: int) -> List[str]:
    """
    Merge small adjacent chunks to avoid orphan fragments.

    Playbook rule:
    - Min chunk size: 100 chars
    - Merge chunks under 200 chars with the next chunk
    """
    if min_merge_chars <= 0 or not chunks:
        return chunks

    merged: List[str] = []
    buf: List[str] = []
    buf_len = 0

    for ch in chunks:
        if not buf:
            buf = [ch]
            buf_len = len(ch)
            continue

        if buf_len < min_merge_chars:
            buf.append(ch)
            buf_len += 2 + len(ch)
            continue

        merged.append("\n\n".join(buf).strip())
        buf = [ch]
        buf_len = len(ch)

    if buf:
        merged.append("\n\n".join(buf).strip())

    return merged


@dataclass(frozen=True)
class Chunker:
    # Defaults remain conservative for callers using Chunker directly.
    # ContextEngineConfig sets stronger defaults for production usage.
    min_chars: int = 40
    merge_under_chars: int = 0

    def chunk(self, context: Union[str, Sequence[str]]) -> List[str]:
        from_string = isinstance(context, str)
        if from_string:
            raw = [c for c in _para_split_re.split(context) if c.strip()]
        else:
            # Respect caller-provided boundaries; do not merge list inputs.
            raw = list(context)

        out: List[str] = []
        for c in raw:
            cc = _clean_chunk(str(c))
            if not cc:
                continue
            # For string inputs, we will merge small fragments up to `min_chars`.
            # Dropping fragments outright is risky (can delete critical info).
            if from_string:
                out.append(cc)
            else:
                # For list inputs, caller already chunked; never drop based on size.
                out.append(cc)

        if not from_string:
            return out

        # Merge up to reach `min_chars` coherence, then apply the secondary
        # "merge-under" rule to eliminate small orphans.
        if self.min_chars > 0:
            merged_min: List[str] = []
            buf: List[str] = []
            buf_len = 0
            for ch in out:
                if not buf:
                    buf = [ch]
                    buf_len = len(ch)
                    continue
                if buf_len < self.min_chars:
                    buf.append(ch)
                    buf_len += 2 + len(ch)
                    continue
                merged_min.append("\n\n".join(buf).strip())
                buf = [ch]
                buf_len = len(ch)
            if buf:
                flushed = "\n\n".join(buf).strip()
                if buf_len >= self.min_chars or not merged_min:
                    merged_min.append(flushed)
                else:
                    merged_min[-1] = merged_min[-1] + "\n\n" + flushed
            out = merged_min

        if self.merge_under_chars > 0:
            out = _merge_small_chunks(out, min_merge_chars=self.merge_under_chars)
        return out


_LEGAL_HEADER_RE = re.compile(
    r"^\s*(?:"
    r"(?:ARTICLE|Article)\s+[IVXLC0-9]+[.: -]?"          # ARTICLE IV
    r"|(?:SECTION|Section)\s+\d+(?:\.\d+)*[.: -]?"      # Section 4.2
    r"|(?:CLAUSE|Clause)\s+\d+(?:\.\d+)*[.: -]?"        # Clause 12
    r"|\d+(?:\.\d+)+[)\.]?\s+"                          # 4.2 / 4.2.1
    r"|\d+\)\s+"                                        # 1)
    r")",
    re.MULTILINE,
)


def _looks_like_legal(text: str) -> bool:
    # Heuristic: multiple section/clause headers and dense text.
    if not text:
        return False
    headers = len(_LEGAL_HEADER_RE.findall(text))
    return headers >= 4

_PY_CODE_DEF_RE = re.compile(r"^\s*(def|class)\s+[A-Za-z_]\w*\b", re.MULTILINE)


def _looks_like_python_code(text: str) -> bool:
    """
    Heuristic to detect Python source text when doc_type='auto'.

    We keep this conservative to avoid misclassifying prose with 'def'/'class'.
    """
    if not text:
        return False
    # def/class + indentation suggests code.
    defs = len(_PY_CODE_DEF_RE.findall(text))
    if defs < 1:
        return False
    lines = text.splitlines()
    if not lines:
        return False
    indented = sum(1 for ln in lines if ln.startswith(("    ", "\t")))
    return indented >= 2


@dataclass(frozen=True)
class SmartChunker:
    """
    Document-aware chunker (zero-dep).

    Goals for contracts/legal docs:
    - Group each section/clause header with its body
    - Avoid page-wise chunks (PDF artifacts)
    - Produce coherent chunks sized for retrieval + compression
    """

    min_chars: int = 100
    merge_under_chars: int = 200
    legal_target_chars: int = 1800  # roughly ~450 tokens under heuristic tokenizer

    def chunk(self, text: str, *, doc_type: str = "auto") -> List[str]:
        dt = (doc_type or "auto").lower()
        if dt == "code":
            return self._chunk_python(text)
        if dt in ("legal", "contract"):
            return self._chunk_legal(text)
        if dt == "pdf":
            # PDFs often have messy line breaks. Normalize and then chunk.
            normalized = self._normalize_pdf_text(text)
            if _looks_like_legal(normalized):
                return self._chunk_legal(normalized)
            return Chunker(min_chars=self.min_chars, merge_under_chars=self.merge_under_chars).chunk(normalized)

        if dt == "auto" and _looks_like_legal(text):
            return self._chunk_legal(text)
        if dt == "auto" and _looks_like_python_code(text):
            return self._chunk_python(text)

        chunks = Chunker(min_chars=self.min_chars, merge_under_chars=self.merge_under_chars).chunk(text)
        if not chunks and text.strip():
            return [text.strip()]
        return chunks

    def _chunk_python(self, text: str) -> List[str]:
        t = text.rstrip("\n")
        if not t.strip():
            return []

        lines = t.splitlines()

        # Python-only boundaries: keep decorators with the def/class they apply to.
        py_def = re.compile(r"^(def|class)\s+[A-Za-z_]\w*\b")

        def is_toplevel(ln: str) -> bool:
            return not ln.startswith((" ", "\t"))

        def is_boundary(ln: str) -> bool:
            s = ln.strip()
            if not s or s.startswith("#"):
                return False
            if not is_toplevel(ln):
                return False
            if s.startswith(("import ", "from ")):
                # Start/continue the import region at top-level.
                return True
            if s.startswith("@"):
                # Decorators belong with the following def/class.
                return True
            return bool(py_def.match(s))

        blocks: List[str] = []
        cur: List[str] = []

        def flush() -> None:
            if not cur:
                return
            b = "\n".join(cur).rstrip()
            if b.strip():
                blocks.append(b)

        for ln in lines:
            if is_boundary(ln) and cur:
                flush()
                cur = [ln]
            else:
                cur.append(ln)
        flush()

        # Split off huge trailing comment tails from code blocks.
        # This prevents large comment dumps from diluting relevance scoring
        # for the function/class definitions above.
        split_blocks: List[str] = []
        for b in blocks:
            bl = b.splitlines()
            if len(bl) < 120:
                split_blocks.append(b)
                continue

            # Count trailing top-level-ish comment lines.
            i = len(bl) - 1
            comment_run = 0
            while i >= 0:
                s = bl[i].lstrip()
                if s.startswith("#") and not bl[i].startswith(("    ", "\t")):
                    comment_run += 1
                    i -= 1
                    continue
                break

            if comment_run >= 80 and i >= 0:
                main = "\n".join(bl[: i + 1]).rstrip()
                tail = "\n".join(bl[i + 1 :]).rstrip()
                if main.strip():
                    split_blocks.append(main)
                if tail.strip():
                    split_blocks.append(tail)
            else:
                split_blocks.append(b)

        blocks = split_blocks

        # Pack blocks into target size; never split a block mid-function/class.
        target = max(800, int(self.legal_target_chars))
        chunks: List[str] = []
        buf: List[str] = []
        buf_len = 0
        for b in blocks:
            if not buf:
                buf = [b]
                buf_len = len(b)
                continue
            prospective = buf_len + 2 + len(b)
            if buf_len < target and prospective <= int(target * 1.15):
                buf.append(b)
                buf_len = prospective
                continue
            chunks.append("\n\n".join(buf).strip())
            buf = [b]
            buf_len = len(b)
        if buf:
            chunks.append("\n\n".join(buf).strip())

        # Guarantee at least one chunk for non-empty input.
        return chunks or [t.strip()]

    def _normalize_pdf_text(self, text: str) -> str:
        # Collapse excessive single newlines into spaces while preserving paragraph breaks.
        # This is intentionally conservative: we keep double-newlines as paragraph boundaries.
        lines = text.splitlines()
        out: List[str] = []
        blank_run = 0
        for ln in lines:
            s = ln.strip()
            if not s:
                blank_run += 1
                if blank_run == 1:
                    out.append("")  # represent paragraph break
                continue
            blank_run = 0
            out.append(s)
        # Join: empty string becomes blank line; other lines become space-joined paragraphs.
        buf: List[str] = []
        paras: List[str] = []
        for item in out:
            if item == "":
                if buf:
                    paras.append(" ".join(buf).strip())
                    buf = []
            else:
                buf.append(item)
        if buf:
            paras.append(" ".join(buf).strip())
        return "\n\n".join(p for p in paras if p)

    def _chunk_legal(self, text: str) -> List[str]:
        t = text.strip()
        if not t:
            return []

        # Split on legal headers; keep headers with their bodies.
        lines = [ln.rstrip() for ln in t.splitlines()]
        blocks: List[str] = []
        cur: List[str] = []

        def flush() -> None:
            if cur:
                s = "\n".join(cur).strip()
                if s:
                    blocks.append(s)

        for ln in lines:
            if _LEGAL_HEADER_RE.match(ln):
                flush()
                cur = [ln]
            else:
                cur.append(ln)
        flush()

        # Now merge blocks into target-sized chunks, but never split a block.
        chunks: List[str] = []
        buf: List[str] = []
        buf_len = 0
        target = max(400, int(self.legal_target_chars))

        for b in blocks:
            if not buf:
                buf = [b]
                buf_len = len(b)
                continue
            if buf_len < target:
                buf.append(b)
                buf_len += 2 + len(b)
                continue
            chunks.append("\n\n".join(buf).strip())
            buf = [b]
            buf_len = len(b)

        if buf:
            chunks.append("\n\n".join(buf).strip())

        # Final pass: apply general coherence rules.
        # (These merges help with small “hanging” clause fragments.)
        chunks = Chunker(min_chars=self.min_chars, merge_under_chars=self.merge_under_chars).chunk("\n\n".join(chunks))
        return chunks

