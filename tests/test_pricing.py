from contextbuddy.pricing import get_pricing, PRESETS, OPENAI_GPT4O_MINI


def test_exact_match() -> None:
    p = get_pricing("gpt-4o")
    assert p.input_per_1k > 0


def test_fuzzy_match() -> None:
    p = get_pricing("claude-sonnet-4-20260514")
    assert p.input_per_1k > 0


def test_fallback() -> None:
    p = get_pricing("totally-unknown-model")
    assert p == OPENAI_GPT4O_MINI


def test_all_presets_have_positive_input_cost() -> None:
    for name, p in PRESETS.items():
        if name == "local":
            continue
        assert p.input_per_1k > 0, f"{name} should have positive input cost"
