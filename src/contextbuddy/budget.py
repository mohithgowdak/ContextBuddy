from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

from .types import Tokenizer


_sent_re = re.compile(r"(?<=[.!?])\s+")


def extractive_summarize(text: str, *, max_chars: int) -> str:
    """
    Sentence-safe summarizer: keeps leading sentences until max_chars.
    Deterministic, dependency-free, and never cuts mid-sentence — the
    loop bounds `total` before appending, so the assembled output is
    always <= max_chars without needing an unsafe tail slice.
    """
    if max_chars <= 0:
        return ""
    sents = _sent_re.split(text.strip())
    out: List[str] = []
    total = 0
    for s in sents:
        s = s.strip()
        if not s:
            continue
        add = ("" if not out else " ") + s
        if total + len(add) > max_chars and out:
            break
        out.append(s)
        total += len(add)
        if total >= max_chars:
            break
    return " ".join(out).strip()


@dataclass(frozen=True)
class BudgetEnforcer:
    tokenizer: Tokenizer

    def count_tokens(self, text: str) -> int:
        return self.tokenizer.count_tokens(text)

    def enforce(
        self,
        *,
        chunks: Sequence[str],
        scores: Sequence[float],
        keep_mask: Sequence[bool],
        max_context_tokens: int,
        summary_max_chars: int = 1200,
    ) -> Tuple[List[str], List[int]]:
        """
        Returns (selected_chunks, selected_indices_in_original).
        """
        if max_context_tokens <= 0:
            return [], []

        indexed = list(range(len(chunks)))
        # Keep-chunks first, then by score desc.
        indexed.sort(key=lambda i: (0 if keep_mask[i] else 1, -float(scores[i])))

        # If the keep-set alone can't fit, fall back to a single sentence-safe
        # summary of the keep-set. This preserves entity context under extreme
        # budgets instead of silently dropping "must keep" chunks.
        keep_indices = [i for i in range(len(chunks)) if keep_mask[i]]
        if keep_indices:
            keep_text = "\n\n".join(chunks[i] for i in keep_indices).strip()
            if keep_text and self.count_tokens(keep_text) > max_context_tokens:
                max_chars = min(summary_max_chars, max(80, int(max_context_tokens * 4)))
                summarized = extractive_summarize(keep_text, max_chars=max_chars)
                char_budget = max_chars
                while summarized and self.count_tokens(summarized) > max_context_tokens:
                    char_budget = int(char_budget * 0.8)
                    if char_budget < 40:
                        summarized = ""
                        break
                    summarized = extractive_summarize(keep_text, max_chars=char_budget)
                if summarized:
                    # Index doesn't map cleanly; return a synthetic single chunk.
                    return [summarized], [keep_indices[0]]

        selected: List[int] = []
        total = 0

        for i in indexed:
            t = self.count_tokens(chunks[i])
            if total + t <= max_context_tokens:
                selected.append(i)
                total += t

        if not selected:
            # If nothing fits, summarize the best keep chunk or the best scored chunk.
            best = indexed[0] if indexed else None
            if best is None:
                return [], []
            # Convert token budget to a conservative character budget.
            # This is a best-effort guardrail even with heuristic tokenizers.
            max_chars = min(summary_max_chars, max(80, int(max_context_tokens * 4)))
            summarized = extractive_summarize(chunks[best], max_chars=max_chars)
            # Sentence-safe shrink: if still over budget, re-summarize at a
            # smaller char budget. Never slice mid-sentence.
            char_budget = max_chars
            while summarized and self.count_tokens(summarized) > max_context_tokens:
                char_budget = int(char_budget * 0.8)
                if char_budget < 40:
                    summarized = ""
                    break
                summarized = extractive_summarize(chunks[best], max_chars=char_budget)
            if summarized:
                return [summarized], [best]
            return [], []

        # If we still exceed budget due to token estimator variance, trim tail by score (non-keep first).
        running_total = sum(self.count_tokens(chunks[i]) for i in selected)

        while selected and running_total > max_context_tokens:
            drop_idx = min(
                selected,
                key=lambda i: (1 if keep_mask[i] else 0, float(scores[i])),
            )
            running_total -= self.count_tokens(chunks[drop_idx])
            selected.remove(drop_idx)

        # If budget is extremely tight, compress the last chunk.
        if selected and running_total > max_context_tokens:
            last = selected[-1]
            room = max(0, max_context_tokens - (running_total - self.count_tokens(chunks[last])))
            # Convert token room into chars conservatively.
            max_chars = max(200, int(room * 4))
            chunks2 = list(chunks)
            chunks2[last] = extractive_summarize(chunks[last], max_chars=max_chars)
            return [chunks2[i] for i in selected], selected

        return [chunks[i] for i in selected], selected

