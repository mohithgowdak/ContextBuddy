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
    # The entity keep-list is driven by entities in the *user prompt*.
    final_prompt, report = engine.build_prompt(user_prompt="alpha question about INV-92831", context=context)

    assert "INV-92831" in final_prompt
    assert "KeyEntities:" in final_prompt
    assert any("INV-92831" == e for e in report.entities)


def test_context_entities_do_not_hijack_prompt_entity_keep_list() -> None:
    """
    Entities found in top chunks (e.g. header emails) should be reported,
    but should not force-keep irrelevant chunks unless the user prompt
    contains that entity.
    """
    engine = ContextEngine(
        ContextEngineConfig(max_context_tokens=500, min_relevance=0.20),
    )
    context = (
        "Title page header. Contact: savithan@jssstuniv.in\n\n"
        "Abstract: this is general background.\n\n"
        "METHOD: The procedure is as follows: Step 1 preprocess. Step 2 compute TV-L1. Step 3 HTNet.\n\n"
        "RESULTS: Accuracy 92.4% and MCC 0.8082.\n"
    )
    final_prompt, report = engine.build_prompt(
        user_prompt="explain the procedure of the working method",
        context=context,
    )
    assert "savithan@jssstuniv.in" not in report.entities
    assert "procedure is as follows" in final_prompt.lower()


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

