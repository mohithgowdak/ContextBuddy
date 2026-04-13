from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from contextbuddy import ContextEngine, ContextEngineConfig


@dataclass(frozen=True)
class DummyEmbedder:
    """
    Deterministic embedder for tests.

    Produces a 2D vector:
    - dim0 = count of the word 'alpha'
    - dim1 = count of the word 'beta'
    """

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        out: List[List[float]] = []
        for t in texts:
            tl = t.lower()
            out.append([float(tl.count("alpha")), float(tl.count("beta"))])
        return out


def test_prunes_irrelevant_chunks() -> None:
    engine = ContextEngine(
        ContextEngineConfig(max_context_tokens=10_000, min_relevance=0.2),
        embedder=DummyEmbedder(),
    )

    context = [
        "alpha alpha alpha " * 10,  # relevant
        "completely unrelated text " * 10,  # irrelevant
    ]
    final_prompt, report = engine.build_prompt(user_prompt="alpha question", context=context)

    assert report.total_chunks == 2
    assert report.kept_chunks == 1
    assert "alpha alpha alpha" in final_prompt
    assert "completely unrelated" not in final_prompt


def test_entity_keep_list_prevents_drop() -> None:
    engine = ContextEngine(
        ContextEngineConfig(max_context_tokens=10_000, min_relevance=0.9, include_entities_section=True),
        embedder=DummyEmbedder(),
    )

    context = [
        ("This looks irrelevant but contains invoice_id=INV-92831. " * 5),
        ("alpha alpha alpha " * 10),
    ]
    final_prompt, report = engine.build_prompt(user_prompt="alpha question", context=context)

    assert "INV-92831" in final_prompt
    assert "KeyEntities:" in final_prompt
    assert any("INV-92831" == e for e in report.entities)


def test_budgeting_enforces_max_context_tokens() -> None:
    # Use a tiny budget so only one chunk fits.
    engine = ContextEngine(
        ContextEngineConfig(max_context_tokens=5, min_relevance=0.0),
        embedder=DummyEmbedder(),
    )

    context = [
        "alpha " * 200,
        "beta " * 200,
    ]
    final_prompt, report = engine.build_prompt(user_prompt="alpha", context=context)

    assert report.final_context_tokens <= engine.config.max_context_tokens
    assert report.kept_chunks >= 0

