# Databricks notebook source
# MAGIC %md
# MAGIC # 03 — Train At-Risk Model and Register in Model Registry
# MAGIC
# MAGIC Trains the learner at-risk prediction model on synthetic student data,
# MAGIC logs to Databricks MLflow, and registers in Unity Catalog Model Registry.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Install Dependencies

# COMMAND ----------

# MAGIC %pip install scikit-learn shap
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Configuration

# COMMAND ----------

import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

CATALOG = "medical_education_rag_dbx"
SCHEMA = "rag_data"
REGISTERED_MODEL_NAME = f"{CATALOG}.{SCHEMA}.learner_at_risk_classifier"

mlflow.set_experiment(f"/Users/chris_weinreich@yahoo.com/at_risk_prediction")

print(f"Model will be registered as: {REGISTERED_MODEL_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Generate Synthetic Student Data

# COMMAND ----------

np.random.seed(42)
n_students = 500

data = pd.DataFrame({
    "quiz_scores": np.random.normal(75, 15, n_students).clip(0, 100),
    "attendance_rate": np.random.beta(8, 2, n_students),
    "study_hours_per_week": np.random.exponential(10, n_students).clip(0, 40),
    "forum_posts": np.random.poisson(5, n_students),
    "assignment_completion": np.random.beta(7, 3, n_students),
    "days_since_last_login": np.random.exponential(3, n_students).clip(0, 30),
})

# Generate labels: at-risk based on weighted factors
risk_score = (
    (100 - data["quiz_scores"]) * 0.3
    + (1 - data["attendance_rate"]) * 100 * 0.25
    + (40 - data["study_hours_per_week"]).clip(0) * 0.15
    + data["days_since_last_login"] * 2
    + (1 - data["assignment_completion"]) * 100 * 0.2
)
data["at_risk"] = (risk_score > np.percentile(risk_score, 70)).astype(int)

# Feature engineering
data["engagement_score"] = (
    data["attendance_rate"] * 0.3
    + data["assignment_completion"] * 0.3
    + (data["forum_posts"] / data["forum_posts"].max()) * 0.2
    + (data["study_hours_per_week"] / 40) * 0.2
)
data["quiz_trend"] = data["quiz_scores"] - data["quiz_scores"].mean()
data["attendance_x_quiz"] = data["attendance_rate"] * data["quiz_scores"]

feature_cols = [
    "quiz_scores", "attendance_rate", "study_hours_per_week",
    "forum_posts", "assignment_completion", "days_since_last_login",
    "engagement_score", "quiz_trend", "attendance_x_quiz",
]

X = data[feature_cols].values
y = data["at_risk"].values

print(f"Students: {n_students}, Features: {len(feature_cols)}")
print(f"At-risk: {y.sum()} ({y.mean():.0%}), Not at-risk: {(1-y).sum()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Train Model and Log to MLflow

# COMMAND ----------

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", GradientBoostingClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42,
    )),
])

# Cross-validation
cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="f1")
print(f"F1 scores (5-fold CV): {cv_scores}")
print(f"Mean F1: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

# Train on full data
pipeline.fit(X, y)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Register Model in Unity Catalog

# COMMAND ----------

with mlflow.start_run(run_name="at_risk_gbm") as run:
    # Log parameters
    mlflow.log_params({
        "model_type": "GradientBoostingClassifier",
        "n_estimators": 100,
        "max_depth": 4,
        "learning_rate": 0.1,
        "n_features": len(feature_cols),
        "n_students": n_students,
    })

    # Log metrics
    mlflow.log_metrics({
        "f1_mean": float(cv_scores.mean()),
        "f1_std": float(cv_scores.std()),
    })

    # Log and register model
    mlflow.sklearn.log_model(
        pipeline,
        artifact_path="at_risk_model",
        registered_model_name=REGISTERED_MODEL_NAME,
    )

    print(f"Run ID: {run.info.run_id}")
    print(f"Model registered as: {REGISTERED_MODEL_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Verify Registration

# COMMAND ----------

from mlflow import MlflowClient

client = MlflowClient()
model_info = client.get_registered_model(REGISTERED_MODEL_NAME)
print(f"Model: {model_info.name}")
print(f"Latest version: {model_info.latest_versions}")

# Test inference with the registered model
model_uri = f"models:/{REGISTERED_MODEL_NAME}/1"
loaded_model = mlflow.sklearn.load_model(model_uri)
sample_prediction = loaded_model.predict(X[:5])
print(f"\nSample predictions: {sample_prediction}")
print(f"Sample actuals:     {y[:5]}")
