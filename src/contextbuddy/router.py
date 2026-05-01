from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .pricing import OPENAI_GPT4O, OPENAI_GPT4O_MINI, get_pricing
from .types import ModelPricing


_COMPLEX_KEYWORDS = {
    "analyze", "compare", "contrast", "evaluate", "synthesize", "critique",
    "implications", "consequences", "trade-offs", "tradeoffs", "nuance",
    "legal", "regulatory", "compliance", "architecture", "design",
    "strategy", "recommend", "prioritize", "assess", "debate",
    "multi-step", "reasoning", "chain-of-thought",
    "summarize", "summarise", "summary",
}

_SIMPLE_KEYWORDS = {
    "what is", "define", "list", "name", "when", "where", "how many",
    "yes or no", "true or false", "extract", "find", "lookup",
}

_WORD_RE = re.compile(r"\b[a-z]+\b")


@dataclass(frozen=True)
class RouteRule:
    max_complexity: float
    model: str
    pricing: ModelPricing


def score_complexity(query: str) -> float:
    """
    Score query complexity from 0.0 (trivial) to 1.0 (highly complex).

    Uses offline heuristics: keyword analysis, query length, question structure.
    """
    q = query.lower().strip()
    words = _WORD_RE.findall(q)
    n_words = len(words)

    score = 0.0

    # Length signal: longer queries tend to be more complex.
    if n_words > 50:
        score += 0.25
    elif n_words > 25:
        score += 0.15
    elif n_words > 10:
        score += 0.05

    # Complex keywords signal.
    complex_hits = sum(1 for kw in _COMPLEX_KEYWORDS if kw in q)
    score += min(0.4, complex_hits * 0.1)

    # Simple keywords reduce complexity.
    simple_hits = sum(1 for kw in _SIMPLE_KEYWORDS if kw in q)
    score -= min(0.2, simple_hits * 0.1)

    # Multiple questions signal.
    question_marks = q.count("?")
    if question_marks >= 2:
        score += 0.1

    # Conditional/comparative language.
    if any(w in q for w in ("if", "versus", "vs", "compared to", "difference between")):
        score += 0.1

    # Imperative task verbs on short queries signal non-trivial work.
    # "summarize the PDF" is 3 words but requires deep comprehension.
    _TASK_VERBS = {
        "summarize", "summarise", "explain", "rewrite", "refactor",
        "debug", "review", "optimize", "translate", "convert",
    }
    if n_words <= 12 and any(q.startswith(v) or v in q for v in _TASK_VERBS):
        score += 0.15

    return max(0.0, min(1.0, score))


class Router:
    """
    Smart model selector.

    Routes queries to cheap or expensive models based on complexity.
    All scoring is offline -- zero API calls.
    """

    def __init__(self, rules: Optional[Sequence[RouteRule | Dict]] = None):
        if rules is None:
            rules = [
                RouteRule(max_complexity=0.3, model="gpt-4o-mini", pricing=OPENAI_GPT4O_MINI),
                RouteRule(max_complexity=1.0, model="gpt-4o", pricing=OPENAI_GPT4O),
            ]

        self.rules: List[RouteRule] = []
        for r in rules:
            if isinstance(r, dict):
                p = r.get("pricing")
                if isinstance(p, str):
                    p = get_pricing(p)
                elif p is None:
                    p = get_pricing(r.get("model", "gpt-4o-mini"))
                self.rules.append(RouteRule(
                    max_complexity=r["max_complexity"],
                    model=r["model"],
                    pricing=p,
                ))
            else:
                self.rules.append(r)

        self.rules.sort(key=lambda r: r.max_complexity)

    def select(self, query: str) -> Tuple[str, ModelPricing]:
        """Returns (model_name, pricing) for the given query."""
        c = score_complexity(query)
        for rule in self.rules:
            if c <= rule.max_complexity:
                return rule.model, rule.pricing
        last = self.rules[-1] if self.rules else RouteRule(1.0, "gpt-4o-mini", OPENAI_GPT4O_MINI)
        return last.model, last.pricing
