import pytest

from contextbuddy.hybrid_scorer import HybridScorer


@pytest.fixture
def scorer():
    return HybridScorer()


class TestHybridScorer:
    def test_empty_chunks(self, scorer):
        assert scorer.score(query="hello", chunks=[]) == []

    def test_exact_match_scores_high(self, scorer):
        chunks = [
            "The payment terms require net-30 settlement.",
            "Our vacation policy allows 20 days off per year.",
            "Company picnic is scheduled for next Friday.",
        ]
        scores = scorer.score(query="What are the payment terms?", chunks=chunks)
        assert scores[0] > scores[1]
        assert scores[0] > scores[2]

    def test_synonym_matching(self, scorer):
        """The killer test: 'car' in query should match 'automobile' in chunk."""
        chunks = [
            "The automobile insurance policy covers collision and liability.",
            "The home insurance policy covers fire and flood damage.",
            "Company cafeteria serves lunch from 12pm to 2pm.",
        ]
        scores = scorer.score(query="What is the car insurance?", chunks=chunks)
        assert scores[0] > scores[2], (
            f"Synonym matching failed: 'automobile' chunk ({scores[0]:.3f}) "
            f"should score higher than cafeteria chunk ({scores[2]:.3f})"
        )

    def test_stemming_matching(self, scorer):
        """'payments' in query matches 'payment' in chunk."""
        chunks = [
            "Payment is due within 30 days of invoice date.",
            "The weather forecast predicts rain for the weekend.",
        ]
        scores = scorer.score(query="When are the payments due?", chunks=chunks)
        assert scores[0] > scores[1]

    def test_fuzzy_ngram_matching(self, scorer):
        """'optimize' matches 'optimise' via character n-gram overlap."""
        chunks = [
            "We need to optimise the database queries for better performance.",
            "The annual company retreat will be held in September.",
        ]
        scores = scorer.score(query="How to optimize database queries?", chunks=chunks)
        assert scores[0] > scores[1]

    def test_numeric_results_bonus(self, scorer):
        """Queries asking for numbers should prefer numeric-heavy chunks."""
        chunks = [
            "Abstract: This paper proposes a method and discusses results qualitatively.",
            "Results: Accuracy 92.4% on Dataset-A, F1 0.81, AUC 0.93, FPS 120.",
            "Conclusion: We discuss future work and limitations.",
        ]
        scores = scorer.score(query="Give me the numerical results and accuracy metrics", chunks=chunks)
        assert scores[1] == max(scores)

    def test_method_procedure_bonus(self, scorer):
        """Queries asking for procedure/method should prefer methodology chunks."""
        chunks = [
            "Abstract: this paper proposes a method and gives an overview.",
            "IV. PROPOSED METHODOLOGY The procedure is as follows: Step 1 preprocess. Step 2 extract TV-L1.",
            "RESULTS: Accuracy 92.4% on Dataset-A.",
        ]
        scores = scorer.score(query="explain the procedure of the working method", chunks=chunks)
        assert scores[1] == max(scores)

    def test_proposed_methodology_beats_related_work(self, scorer):
        chunks = [
            "II. RELATED WORKS This methodology is beneficial in prior studies but does not describe our steps.",
            "IV. PROPOSED METHODOLOGY The proposed methodology reflects a complete pipeline. Step 1 preprocess.",
        ]
        scores = scorer.score(query="explain the procedure of the working method", chunks=chunks)
        assert scores[1] > scores[0]

    def test_related_work_intent_prefers_related_work_section(self, scorer):
        chunks = [
            "RESULTS: Accuracy 92.4% and MCC 0.8082. Fig.4 shows UF1 vs heads.",
            "II. RELATED WORKS Hashmi et al. [4] introduce LARNet. Li et al. [5] use attention mechanisms.",
            "IV. PROPOSED METHODOLOGY Step 1 preprocess. Step 2 compute optical flow.",
        ]
        scores = scorer.score(query="find me relevant previous works", chunks=chunks)
        assert scores[1] == max(scores)

    def test_idf_weighting(self, scorer):
        """Rare terms should matter more than common terms."""
        chunks = [
            "The quarterly revenue report shows strong growth in enterprise sales.",
            "The quarterly newsletter mentions company updates and events.",
            "The quarterly team meeting is scheduled for Monday afternoon.",
        ]
        scores = scorer.score(query="What does the revenue report say?", chunks=chunks)
        assert scores[0] == max(scores)

    def test_returns_correct_length(self, scorer):
        chunks = ["chunk one", "chunk two", "chunk three"]
        scores = scorer.score(query="test query", chunks=chunks)
        assert len(scores) == 3

    def test_all_scores_bounded(self, scorer):
        chunks = [
            "First paragraph about various topics.",
            "Second paragraph about different subjects.",
            "Third paragraph about other matters entirely.",
        ]
        scores = scorer.score(query="topics and subjects", chunks=chunks)
        for s in scores:
            assert 0.0 <= s <= 1.0 + 1e-9, f"Score {s} out of bounds"

    def test_single_chunk(self, scorer):
        scores = scorer.score(query="test", chunks=["some text about testing"])
        assert len(scores) == 1
        assert scores[0] >= 0.0

    def test_mixed_signals(self, scorer):
        """Chunk that matches via BM25 + synonym + n-gram should score highest."""
        chunks = [
            "We need to purchase a new automobile for the fleet. "
            "The cost of the vehicle acquisition should be budgeted.",
            "The cafeteria menu has been updated with new lunch options.",
        ]
        scores = scorer.score(
            query="How much does it cost to buy a car?",
            chunks=chunks,
        )
        assert scores[0] > scores[1]


class TestHybridScorerVsOld:
    """
    Regression tests: the HybridScorer should handle cases that the old
    LocalHashEmbedder + cosine similarity would fail on.
    """

    def test_synonym_car_automobile(self, scorer):
        chunks = [
            "The automobile coverage plan includes collision protection.",
            "Employee handbook section on workplace conduct.",
        ]
        scores = scorer.score(query="car insurance coverage", chunks=chunks)
        assert scores[0] > scores[1]

    def test_synonym_buy_purchase(self, scorer):
        chunks = [
            "To purchase equipment, fill out a procurement request form.",
            "The park closes at sunset during winter months.",
        ]
        scores = scorer.score(query="How do I buy new equipment?", chunks=chunks)
        assert scores[0] > scores[1]

    def test_synonym_salary_compensation(self, scorer):
        chunks = [
            "Compensation reviews are conducted annually in March.",
            "The building's fire alarm system was last inspected in January.",
        ]
        scores = scorer.score(query="When is the salary review?", chunks=chunks)
        assert scores[0] > scores[1]

    def test_synonym_error_bug(self, scorer):
        chunks = [
            "The bug was traced to a null pointer in the auth module.",
            "New furniture was ordered for the conference room.",
        ]
        scores = scorer.score(query="What caused the error in authentication?", chunks=chunks)
        assert scores[0] > scores[1]

    def test_synonym_fast_quick(self, scorer):
        chunks = [
            "The quick deployment process takes under 5 minutes.",
            "Annual budget planning begins in Q4.",
        ]
        scores = scorer.score(query="How fast is the deployment?", chunks=chunks)
        assert scores[0] > scores[1]
