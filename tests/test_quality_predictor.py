"""Tests for the retrieval quality predictor."""

import numpy as np

from src.retrieval.quality_predictor import RetrievalQualityPredictor


class TestQualityPredictor:
    """Tests for RetrievalQualityPredictor."""

    def test_extract_features_shape(self):
        """Feature extraction returns correct dimensions (1, 5)."""
        predictor = RetrievalQualityPredictor()
        features = predictor.extract_features(
            cosine_similarity=0.85,
            bm25_score=12.5,
            token_overlap=0.4,
            chunk_length=350,
            qtype_match=1,
        )
        assert features.shape == (1, 5)

    def test_predict_untrained_returns_default(self):
        """Untrained predictor returns 0.5 (neutral)."""
        predictor = RetrievalQualityPredictor()
        features = predictor.extract_features(
            cosine_similarity=0.8,
            bm25_score=10.0,
            token_overlap=0.3,
            chunk_length=200,
            qtype_match=1,
        )
        score = predictor.predict(features)
        assert score == 0.5

    def test_predict_scores_in_range(self):
        """After training, all predictions are between 0.0 and 1.0."""
        predictor = RetrievalQualityPredictor()

        # Generate synthetic training data
        rng = np.random.RandomState(42)
        X = rng.rand(100, 5)
        y = rng.rand(100)

        predictor.pipeline.fit(X, y)
        predictor._is_trained = True

        test_features = rng.rand(10, 5)
        score = predictor.predict(test_features)
        assert 0.0 <= score <= 1.0

    def test_relevant_chunk_scores_higher(self):
        """Relevant chunks score higher than irrelevant ones."""
        predictor = RetrievalQualityPredictor()

        # Train with clear signal: high cosine_sim + high overlap → high score
        rng = np.random.RandomState(42)
        X = rng.rand(200, 5)
        # Target correlated with cosine_similarity (col 0) and token_overlap (col 2)
        y = np.clip(0.5 * X[:, 0] + 0.3 * X[:, 2] + 0.1 * rng.randn(200), 0, 1)

        predictor.pipeline.fit(X, y)
        predictor._is_trained = True

        # Relevant: high cosine_sim, high overlap
        relevant = predictor.extract_features(
            cosine_similarity=0.95, bm25_score=15.0,
            token_overlap=0.8, chunk_length=300, qtype_match=1,
        )
        # Irrelevant: low cosine_sim, low overlap
        irrelevant = predictor.extract_features(
            cosine_similarity=0.1, bm25_score=1.0,
            token_overlap=0.05, chunk_length=300, qtype_match=0,
        )

        score_relevant = predictor.predict(relevant)
        score_irrelevant = predictor.predict(irrelevant)
        assert score_relevant > score_irrelevant
