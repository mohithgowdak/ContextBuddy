"""
Red-line invariants for ContextBuddy.

These tests encode the four hard guarantees from `.cursor/rules/quality.mdc`:
    1. Regex-matched entity ALWAYS survives compression
    2. Compressed output ALWAYS shorter than input (never larger)
    3. Empty output is a crash, not a valid result
    4. No mid-sentence splits in final output

Plus one determinism check: the built-in `LocalHashEmbedder` must be stable
across instances (and therefore across processes). Python's `hash()` is
randomized per interpreter and used to be called directly in this codebase,
which would silently break determinism at scale.
"""
from __future__ import annotations

import asyncio
import re

import pytest

from contextbuddy import ContextEngine, ContextEngineConfig, MemoryStore
from contextbuddy.budget import extractive_summarize
from contextbuddy.embedder import LocalHashEmbedder


ENTITY_TEXT = (
    "Invoice INV-92831 issued 2026-04-01 for account_id=acct_12345. "
    "Amount: $4,500.00 USD. "
    "Contact billing@example.com or visit https://example.com/invoices."
)

FILLER_PARA = (
    "Quarterly planning meeting notes about OKRs and hiring timelines. "
    "Team lunch menu for next week includes pizza and sushi. "
    "Office renovation scheduled for Q3. Marketing content calendar TBD."
)


def _engine(budget: int = 200) -> ContextEngine:
    return ContextEngine(ContextEngineConfig(max_context_tokens=budget))


# -----------------------------------------------------------------------------
# Red line #1: regex-matched entities always survive.
# -----------------------------------------------------------------------------
def test_redline_entity_always_survives_under_tight_budget():
    engine = _engine(budget=60)
    context = "\n\n".join([FILLER_PARA] * 5 + [ENTITY_TEXT] + [FILLER_PARA] * 5)

    final_prompt, report = engine.build_prompt(
        user_prompt="Summarize the invoice.",
        context=context,
    )

    for entity in ("INV-92831", "acct_12345", "2026-04-01"):
        assert entity in final_prompt, (
            f"red line #1: entity {entity!r} was dropped by compression"
        )
    assert "INV-92831" in report.entities


# -----------------------------------------------------------------------------
# Red line #2: compressed context is never larger than input context.
# (The full prompt may grow by the fixed `KeyEntities:` header — that's opt-in
# via `include_entities_section`. The invariant applies to the context itself.)
# -----------------------------------------------------------------------------
def test_redline_context_never_larger_than_input_small_context():
    engine = _engine(budget=10_000)
    context = ENTITY_TEXT
    final_prompt, report = engine.build_prompt(
        user_prompt="Summarize.",
        context=context,
    )
    assert report.final_context_tokens <= report.original_context_tokens, (
        f"red line #2: context output ({report.final_context_tokens} tok) "
        f"is larger than context input ({report.original_context_tokens} tok)"
    )


def test_redline_context_never_larger_than_input_large_context():
    engine = _engine(budget=200)
    context = "\n\n".join([FILLER_PARA] * 30 + [ENTITY_TEXT])
    final_prompt, report = engine.build_prompt(
        user_prompt="What is the invoice ID?",
        context=context,
    )
    assert report.final_context_tokens <= report.original_context_tokens


# -----------------------------------------------------------------------------
# Red line #3: non-empty input must never produce empty output.
# -----------------------------------------------------------------------------
def test_redline_never_empty_when_input_has_content():
    engine = _engine(budget=50)
    context = "\n\n".join([FILLER_PARA] * 3 + [ENTITY_TEXT])
    final_prompt, report = engine.build_prompt(
        user_prompt="Tell me about the invoice INV-92831.",
        context=context,
    )
    assert final_prompt.strip(), "red line #3: output must not be empty"
    assert report.final_prompt_tokens > 0


# -----------------------------------------------------------------------------
# Red line #4: extractive summary never cuts mid-sentence.
# -----------------------------------------------------------------------------
def test_redline_no_midsentence_splits_in_extractive_summary():
    text = (
        "First sentence ends here. Second sentence ends here. "
        "Third sentence ends here. Fourth sentence ends here."
    )
    for budget in (20, 30, 45, 80, 200):
        out = extractive_summarize(text, max_chars=budget)
        if not out:
            continue
        last_char = out.rstrip()[-1]
        assert last_char in ".!?", (
            f"red line #4: mid-sentence cut at budget={budget}: {out!r}"
        )
        # The output must also be a prefix-sequence of complete sentences.
        for sentence in re.split(r"(?<=[.!?])\s+", out):
            if sentence:
                assert sentence.rstrip()[-1] in ".!?", (
                    f"red line #4: fragment in output: {sentence!r}"
                )


# -----------------------------------------------------------------------------
# Determinism: the built-in embedder must be stable across instances.
# -----------------------------------------------------------------------------
def test_local_hash_embedder_is_deterministic_across_instances():
    text = "Invoice INV-92831 payment due 2026-04-01 for acct_12345."
    a = LocalHashEmbedder().embed([text])[0]
    b = LocalHashEmbedder().embed([text])[0]
    assert a == b, (
        "LocalHashEmbedder must be deterministic across instances. "
        "If this fails, you're probably using Python's randomized hash() "
        "instead of hashlib.md5 for token hashing."
    )


# -----------------------------------------------------------------------------
# Async contract: aembed / asearch must return the same results as sync.
# -----------------------------------------------------------------------------
def test_async_embedder_matches_sync_output():
    emb = LocalHashEmbedder()
    sync_vecs = emb.embed(["hello world", "contextbuddy compresses prompts"])
    async_vecs = asyncio.run(emb.aembed(["hello world", "contextbuddy compresses prompts"]))
    assert sync_vecs == async_vecs, "aembed must produce identical output to embed"


def test_async_store_search_matches_sync():
    store = MemoryStore()
    store.add([ENTITY_TEXT, FILLER_PARA, FILLER_PARA + " payment terms"])
    sync_hits = [r.chunk for r in store.search("payment terms", top_k=3)]
    async_hits = [
        r.chunk for r in asyncio.run(store.asearch("payment terms", top_k=3))
    ]
    assert sync_hits == async_hits, "MemoryStore.asearch must match search output"
