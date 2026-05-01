from __future__ import annotations

from contextbuddy import ContextEngine, ContextEngineConfig


def test_playbook_chunking_min_100_and_merge_under_200() -> None:
    # Two small chunks should be merged so we don't end up with orphan fragments.
    # Chunker filters <100 chars, then merges <200 with next.
    engine = ContextEngine(ContextEngineConfig(max_context_tokens=500))

    # Construct context that yields multiple borderline chunks.
    c1 = "A" * 120
    c2 = "B" * 120
    context = f"{c1}\n\n{c2}"

    prompt, report = engine.build_prompt(user_prompt="What is this?", context=context)
    # Both should survive and be adjacent; merging should keep coherence.
    assert "A" * 50 in prompt
    assert "B" * 50 in prompt


def test_playbook_entity_keeps_neighbor_chunks() -> None:
    engine = ContextEngine(ContextEngineConfig(max_context_tokens=120, min_relevance=0.9))

    before = "Context line before entity chunk. " * 6
    entity = "Invoice INV-92831 issued 2026-04-01 for account_id=acct_12345. " * 3
    after = "Context line after entity chunk. " * 6
    noise = "Completely irrelevant planning notes. " * 20
    context = "\n\n".join([noise, before, entity, after, noise])

    prompt, _ = engine.build_prompt(user_prompt="alpha question", context=context)

    # Entity must survive
    assert "INV-92831" in prompt
    # Neighbor context should also survive (playbook rule)
    assert "before entity chunk" in prompt
    assert "after entity chunk" in prompt


def test_playbook_conservative_mode_lowers_threshold() -> None:
    cfg = ContextEngineConfig(conservative_mode=True, min_relevance=0.15)
    engine = ContextEngine(cfg)
    assert engine.config.min_relevance <= 0.05

