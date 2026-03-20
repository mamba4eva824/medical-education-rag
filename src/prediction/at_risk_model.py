"""At-risk learner prediction pipeline.

Trains a GradientBoostingClassifier on synthetic student engagement data
to predict which learners are at risk of falling behind.
"""

import logging

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class AtRiskPipeline:
    """Predicts at-risk learners from engagement features."""

    FEATURE_NAMES = [
        "quiz_avg",
        "assignment_completion",
        "login_frequency",
        "forum_posts",
        "time_on_platform_min",
        "content_diversity",
    ]

    def __init__(self):
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", GradientBoostingClassifier(
                n_estimators=100, max_depth=4, learning_rate=0.1,
            )),
        ])
        self._is_trained = False

    def create_features(
        self, n_students: int = 500, seed: int = 42,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate synthetic student engagement data.

        Returns:
            (X, y) where X has shape (n_students, 6) and y is binary labels.
        """
        np.random.seed(seed)
        X = np.column_stack([
            np.random.normal(0.7, 0.15, n_students),    # quiz_avg
            np.random.normal(0.75, 0.2, n_students),    # assignment_completion
            np.random.poisson(5, n_students),            # login_frequency
            np.random.poisson(3, n_students),            # forum_posts
            np.random.normal(120, 40, n_students),       # time_on_platform_min
            np.random.uniform(0, 1, n_students),         # content_diversity
        ])

        # Risk score based on weighted combination of low engagement signals
        risk_score = (
            0.4 * (1 - X[:, 0])
            + 0.3 * (1 - X[:, 1])
            + 0.15 * (1 - X[:, 2] / 10)
            + 0.15 * (1 - X[:, 3] / 10)
        )
        y = (risk_score > np.random.normal(0.5, 0.1, n_students)).astype(int)
        return X, y

    def train_and_log(
        self,
        X: np.ndarray,
        y: np.ndarray,
        experiment_name: str = "at_risk_prediction",
    ) -> None:
        """Train the classifier and log results to MLflow."""
        import mlflow
        from sklearn.metrics import f1_score
        from sklearn.model_selection import cross_val_score

        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name="at_risk_gbm_v1"):
            cv_scores = cross_val_score(self.pipeline, X, y, cv=5, scoring="f1")
            self.pipeline.fit(X, y)
            self._is_trained = True

            mlflow.log_params({
                "model_type": "GradientBoostingClassifier",
                "n_estimators": 100,
                "max_depth": 4,
                "learning_rate": 0.1,
                "n_features": X.shape[1],
                "n_students": X.shape[0],
            })
            mlflow.log_metrics({
                "f1_mean_cv5": float(cv_scores.mean()),
                "f1_std_cv5": float(cv_scores.std()),
                "train_f1": float(f1_score(y, self.pipeline.predict(X))),
            })
            mlflow.sklearn.log_model(self.pipeline, "at_risk_model")
            logger.info(
                f"At-risk model trained. F1 CV mean={cv_scores.mean():.4f}"
            )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary at-risk labels."""
        if not self._is_trained:
            raise RuntimeError("Model not trained yet")
        return self.pipeline.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict at-risk probabilities."""
        if not self._is_trained:
            raise RuntimeError("Model not trained yet")
        return self.pipeline.predict_proba(X)
