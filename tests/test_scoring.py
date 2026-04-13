from contextbuddy.scoring import SemanticScorer, cosine_similarity
from contextbuddy.embedder import LocalHashEmbedder


def test_cosine_identical_vectors() -> None:
    v = [1.0, 2.0, 3.0]
    assert abs(cosine_similarity(v, v) - 1.0) < 1e-6


def test_cosine_orthogonal_vectors() -> None:
    a = [1.0, 0.0]
    b = [0.0, 1.0]
    assert abs(cosine_similarity(a, b)) < 1e-6


def test_cosine_opposite_vectors() -> None:
    a = [1.0, 0.0]
    b = [-1.0, 0.0]
    assert abs(cosine_similarity(a, b) - (-1.0)) < 1e-6


def test_scorer_returns_correct_count() -> None:
    scorer = SemanticScorer(embedder=LocalHashEmbedder(dims=64))
    scores = scorer.score(query="hello world", chunks=["hello world", "foo bar", "goodbye"])
    assert len(scores) == 3


def test_scorer_relevant_chunk_scores_higher() -> None:
    scorer = SemanticScorer(embedder=LocalHashEmbedder(dims=128))
    scores = scorer.score(
        query="machine learning algorithms",
        chunks=[
            "machine learning algorithms and neural networks",
            "recipe for chocolate cake with frosting",
        ],
    )
    assert scores[0] > scores[1]


def test_scorer_empty_chunks() -> None:
    scorer = SemanticScorer(embedder=LocalHashEmbedder())
    assert scorer.score(query="test", chunks=[]) == []
