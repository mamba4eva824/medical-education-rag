"""Tests for the retrieval quality predictor."""

import sys
from unittest.mock import MagicMock, patch

import numpy as np

from src.retrieval.quality_predictor import RetrievalQualityPredictor


class TestRetrievalQualityPredictor:
    def test_extract_features_shape(self):
        predictor = RetrievalQualityPredictor()
        features = predictor.extract_features(
            cosine_similarity=0.8,
            bm25_score=1.5,
            token_overlap=0.3,
            chunk_length=200,
            qtype_match=1,
        )
        assert features.shape == (1, 5)

    def test_predict_untrained_returns_default(self):
        predictor = RetrievalQualityPredictor()
        features = np.array([[0.8, 1.5, 0.3, 200, 1]])
        score = predictor.predict(features)
        assert score == 0.5

    def test_feature_names_count(self):
        assert len(RetrievalQualityPredictor.FEATURE_NAMES) == 5
        assert "cosine_similarity" in RetrievalQualityPredictor.FEATURE_NAMES
        assert "bm25_score" in RetrievalQualityPredictor.FEATURE_NAMES

    def test_train_and_log_sets_trained(self):
        # Mock mlflow at the sys.modules level since it's imported locally
        mock_mlflow = MagicMock()
        mock_mlflow.start_run.return_value.__enter__ = MagicMock()
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        with patch.dict(sys.modules, {"mlflow": mock_mlflow, "mlflow.sklearn": mock_mlflow.sklearn}):
            predictor = RetrievalQualityPredictor()
            X = np.random.rand(100, 5)
            y = np.random.rand(100)
            predictor.train_and_log(X, y)

            assert predictor._is_trained is True
            mock_mlflow.set_experiment.assert_called_once()

    def test_predict_after_training(self):
        mock_mlflow = MagicMock()
        mock_mlflow.start_run.return_value.__enter__ = MagicMock()
        mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

        with patch.dict(sys.modules, {"mlflow": mock_mlflow, "mlflow.sklearn": mock_mlflow.sklearn}):
            predictor = RetrievalQualityPredictor()
            X = np.random.rand(100, 5)
            y = np.random.rand(100)
            predictor.train_and_log(X, y)

            score = predictor.predict(np.array([[0.8, 1.5, 0.3, 200, 1]]))
            assert 0.0 <= score <= 1.0
