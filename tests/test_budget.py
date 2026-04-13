from contextbuddy.budget import BudgetEnforcer, extractive_summarize
from contextbuddy.tokenizer import HeuristicTokenizer


def test_extractive_summarize_basic() -> None:
    text = "First sentence. Second sentence. Third sentence. Fourth sentence."
    result = extractive_summarize(text, max_chars=40)
    assert len(result) <= 40
    assert "First sentence." in result


def test_extractive_summarize_zero_max() -> None:
    assert extractive_summarize("Hello.", max_chars=0) == ""


def test_budget_keeps_high_score_chunks() -> None:
    enforcer = BudgetEnforcer(tokenizer=HeuristicTokenizer())
    chunks = ["short chunk one", "short chunk two", "short chunk three"]
    scores = [0.9, 0.1, 0.5]
    keep_mask = [False, False, False]

    selected, indices = enforcer.enforce(
        chunks=chunks,
        scores=scores,
        keep_mask=keep_mask,
        max_context_tokens=10000,
    )
    assert len(selected) == 3


def test_budget_respects_keep_mask() -> None:
    enforcer = BudgetEnforcer(tokenizer=HeuristicTokenizer())
    chunks = ["important entity chunk here", "less important chunk here"]
    scores = [0.1, 0.9]
    keep_mask = [True, False]

    selected, indices = enforcer.enforce(
        chunks=chunks,
        scores=scores,
        keep_mask=keep_mask,
        max_context_tokens=8,
    )
    assert any("important entity" in c for c in selected)


def test_budget_zero_tokens() -> None:
    enforcer = BudgetEnforcer(tokenizer=HeuristicTokenizer())
    selected, indices = enforcer.enforce(
        chunks=["hello world chunk"],
        scores=[1.0],
        keep_mask=[False],
        max_context_tokens=0,
    )
    assert selected == []
