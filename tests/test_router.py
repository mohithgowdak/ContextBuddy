from contextbuddy.router import Router, RouteRule, score_complexity
from contextbuddy.pricing import OPENAI_GPT4O, OPENAI_GPT4O_MINI


def test_simple_query_low_complexity() -> None:
    score = score_complexity("What is 2+2?")
    assert score < 0.3


def test_complex_query_high_complexity() -> None:
    score = score_complexity(
        "Analyze the legal implications of the new regulatory framework "
        "and compare the trade-offs between compliance strategies. "
        "Evaluate the long-term consequences for our architecture."
    )
    assert score > 0.3


def test_router_routes_simple_to_cheap() -> None:
    router = Router([
        RouteRule(max_complexity=0.3, model="gpt-4o-mini", pricing=OPENAI_GPT4O_MINI),
        RouteRule(max_complexity=1.0, model="gpt-4o", pricing=OPENAI_GPT4O),
    ])
    model, pricing = router.select("What is the capital of France?")
    assert model == "gpt-4o-mini"


def test_router_routes_complex_to_expensive() -> None:
    router = Router([
        RouteRule(max_complexity=0.3, model="gpt-4o-mini", pricing=OPENAI_GPT4O_MINI),
        RouteRule(max_complexity=1.0, model="gpt-4o", pricing=OPENAI_GPT4O),
    ])
    model, pricing = router.select(
        "Analyze the trade-offs and implications of this regulatory strategy "
        "compared to the alternative compliance architecture."
    )
    assert model == "gpt-4o"


def test_router_dict_rules() -> None:
    router = Router([
        {"max_complexity": 0.3, "model": "gpt-4o-mini"},
        {"max_complexity": 1.0, "model": "gpt-4o"},
    ])
    model, _ = router.select("hello")
    assert model in ("gpt-4o-mini", "gpt-4o")


def test_router_default_rules() -> None:
    router = Router()
    model, pricing = router.select("test query")
    assert model in ("gpt-4o-mini", "gpt-4o")


def test_score_complexity_empty() -> None:
    assert score_complexity("") == 0.0


def test_score_complexity_range() -> None:
    for q in ["hi", "analyze complex legal regulatory implications deeply"]:
        s = score_complexity(q)
        assert 0.0 <= s <= 1.0
