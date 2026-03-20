"""Retrieval quality predictor — predicts relevance score for retrieved chunks.

Skeleton implementation for Phase 3. Training happens in Phase 4 via
notebooks/05b_quality_predictor.ipynb using cross-encoder distillation.
"""

import logging

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class RetrievalQualityPredictor:
    """Predicts a continuous relevance score (0-1) for retrieved chunks."""

    FEATURE_NAMES = [
        "cosine_similarity",
        "bm25_score",
        "token_overlap",
        "chunk_length",
        "qtype_match",
    ]

    def __init__(self):
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", GradientBoostingRegressor(
                n_estimators=100, max_depth=4, learning_rate=0.1,
            )),
        ])
        self._is_trained = False

    def extract_features(
        self,
        cosine_similarity: float,
        bm25_score: float,
        token_overlap: float,
        chunk_length: int,
        qtype_match: int,
    ) -> np.ndarray:
        """Build a feature vector from retrieval signals."""
        return np.array([[
            cosine_similarity,
            bm25_score,
            token_overlap,
            chunk_length,
            qtype_match,
        ]])

    def predict(self, features: np.ndarray) -> float:
        """Predict relevance score (0-1) for a batch of chunks.

        Returns 0.5 (neutral) if the model has not been trained yet.
        """
        if not self._is_trained:
            return 0.5
        scores = self.pipeline.predict(features)
        return float(np.clip(scores, 0.0, 1.0).mean())

    def train_and_log(
        self,
        X: np.ndarray,
        y: np.ndarray,
        experiment_name: str = "retrieval_quality",
    ) -> None:
        """Train the quality predictor and log to MLflow.

        Called in Phase 4 via notebooks/05b_quality_predictor.ipynb.
        """
        import mlflow
        from sklearn.model_selection import cross_val_score

        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name="quality_predictor_v1"):
            scores = cross_val_score(
                self.pipeline, X, y, cv=5, scoring="neg_mean_squared_error",
            )
            self.pipeline.fit(X, y)
            self._is_trained = True

            rmse = float(np.sqrt(-scores.mean()))
            mlflow.log_params({
                "model_type": "GradientBoostingRegressor",
                "n_estimators": 100,
                "max_depth": 4,
                "learning_rate": 0.1,
                "n_features": X.shape[1],
            })
            mlflow.log_metrics({
                "rmse": rmse,
                "r2_mean_cv5": float(
                    cross_val_score(
                        self.pipeline, X, y, cv=5, scoring="r2",
                    ).mean()
                ),
            })
            mlflow.sklearn.log_model(self.pipeline, "quality_predictor")
            logger.info(f"Quality predictor trained. RMSE={rmse:.4f}")
