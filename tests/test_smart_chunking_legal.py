from __future__ import annotations

from contextbuddy.chunking import SmartChunker


def test_smart_chunker_groups_legal_sections() -> None:
    text = """
ARTICLE I - DEFINITIONS
1.1 "Agreement" means this contract.
1.2 "Party" means either signatory.

ARTICLE II - PAYMENT TERMS
2.1 Payment due within 30 days.
2.2 Late fee applies after 45 days.

ARTICLE III - TERMINATION
3.1 Either Party may terminate with 30 days notice.
3.2 Termination for cause is immediate.
""".strip()

    chunks = SmartChunker(legal_target_chars=400).chunk(text, doc_type="legal")

    # Should not be tiny per-line chunks; should keep headers with their bodies.
    assert any("ARTICLE II" in c and "Payment due" in c for c in chunks)
    assert all(len(c) >= 100 for c in chunks)
