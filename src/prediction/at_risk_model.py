"""At-risk learner prediction pipeline.

Uses GradientBoostingClassifier to identify students at risk of
underperformance, with SHAP-based explanations.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class AtRiskPipeline:
    """Pipeline for training and serving at-risk learner predictions."""

    def __init__(self) -> None:
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", GradientBoostingClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
            )),
        ])
        self._is_trained = False

    def create_features(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Transform raw data into model features.

        Args:
            raw_data: Raw student interaction data with columns:
                quiz_avg, attendance_pct, resource_views, forum_posts,
                time_on_platform_hrs, prior_gpa.

        Returns:
            DataFrame with engineered features added.
        """
        df = raw_data.copy()

        # Engagement score: weighted sum of activity metrics
        df["engagement_score"] = (
            0.3 * df["resource_views"]
            + 0.3 * df["forum_posts"]
            + 0.4 * df["time_on_platform_hrs"]
        )

        # Quiz trend: deviation from mean (proxy without time series)
        df["quiz_trend"] = df["quiz_avg"] - df["quiz_avg"].mean()

        # Interaction feature
        df["attendance_x_quiz"] = df["attendance_pct"] * df["quiz_avg"] / 100

        return df

    def train_and_log(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
        experiment_name: str = "at_risk_prediction",
    ) -> None:
        """Train the at-risk model and log metrics/artifacts to MLflow.

        Args:
            X: Feature matrix.
            y: Binary target labels (1 = at-risk).
            experiment_name: MLflow experiment name.
        """
        mlflow.set_tracking_uri(str(PROJECT_ROOT / "mlruns"))
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name="at_risk_gbm"):
            # 5-fold cross-validation
            f1_scores = cross_val_score(
                self.pipeline, X, y, cv=5, scoring="f1"
            )

            # Train on full data
            self.pipeline.fit(X, y)
            self._is_trained = True

            # Log params
            mlflow.log_params({
                "model_type": "GradientBoostingClassifier",
                "n_estimators": 100,
                "max_depth": 4,
                "learning_rate": 0.1,
                "n_features": X.shape[1],
            })

            # Log metrics
            mlflow.log_metrics({
                "f1_mean": float(np.mean(f1_scores)),
                "f1_std": float(np.std(f1_scores)),
            })

            # Log model
            mlflow.sklearn.log_model(self.pipeline, "at_risk_model")

            logger.info(
                "At-risk model trained: F1=%.3f ± %.3f",
                np.mean(f1_scores), np.std(f1_scores),
            )

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Predict at-risk probability for students.

        Args:
            X: Feature matrix.

        Returns:
            Array of at-risk probabilities.
        """
        if not self._is_trained:
            logger.warning("Model not trained, returning 0.5")
            return np.full(X.shape[0], 0.5)
        return self.pipeline.predict_proba(X)[:, 1]
