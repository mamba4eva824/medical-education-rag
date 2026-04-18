# Databricks notebook source
# MAGIC %md
# MAGIC # 03 — Model Registry: All Pipeline Models
# MAGIC
# MAGIC Registers all ML models used in the RAG pipeline to Unity Catalog:
# MAGIC 1. **At-risk learner classifier** — predicts students needing intervention
# MAGIC 2. **PubMedBert embedding model** — encodes medical text for vector search
# MAGIC 3. **Cross-encoder reranker** — two-stage retrieval scoring

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Install Dependencies

# COMMAND ----------

# MAGIC %pip install scikit-learn sentence-transformers
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Configuration

# COMMAND ----------

import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature

CATALOG = "medical_education_rag_dbx"
SCHEMA = "rag_data"

mlflow.set_experiment(f"/Users/chris_weinreich@yahoo.com/model_registry")

print(f"Catalog: {CATALOG}.{SCHEMA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Register At-Risk Learner Classifier

# COMMAND ----------

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Generate synthetic student data
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

risk_score = (
    (100 - data["quiz_scores"]) * 0.3
    + (1 - data["attendance_rate"]) * 100 * 0.25
    + (40 - data["study_hours_per_week"]).clip(0) * 0.15
    + data["days_since_last_login"] * 2
    + (1 - data["assignment_completion"]) * 100 * 0.2
)
data["at_risk"] = (risk_score > np.percentile(risk_score, 70)).astype(int)

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

# Train
sk_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", GradientBoostingClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42,
    )),
])
cv_scores = cross_val_score(sk_pipeline, X, y, cv=5, scoring="f1")
sk_pipeline.fit(X, y)

# Register
ATRISK_MODEL = f"{CATALOG}.{SCHEMA}.learner_at_risk_classifier"

with mlflow.start_run(run_name="at_risk_gbm"):
    mlflow.log_params({
        "model_type": "GradientBoostingClassifier",
        "n_estimators": 100, "max_depth": 4, "learning_rate": 0.1,
        "n_features": len(feature_cols), "n_students": n_students,
    })
    mlflow.log_metrics({"f1_mean": float(cv_scores.mean()), "f1_std": float(cv_scores.std())})

    signature = infer_signature(X, sk_pipeline.predict(X))
    mlflow.sklearn.log_model(
        sk_pipeline, artifact_path="at_risk_model",
        registered_model_name=ATRISK_MODEL, signature=signature,
    )

print(f"Registered: {ATRISK_MODEL}")
print(f"F1: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Register PubMedBert Embedding Model

# COMMAND ----------

from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL_NAME = "pritamdeka/S-PubMedBert-MS-MARCO"
EMBEDDING_REGISTRY = f"{CATALOG}.{SCHEMA}.pubmedbert_embeddings"

embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

# Create signature: string input -> float array output
sample_input = pd.DataFrame({"text": ["What are the symptoms of heart failure?"]})
sample_output = embedding_model.encode(sample_input["text"].tolist())
signature = infer_signature(sample_input, sample_output)

with mlflow.start_run(run_name="pubmedbert_embedding"):
    mlflow.log_params({
        "model_name": EMBEDDING_MODEL_NAME,
        "model_type": "SentenceTransformer",
        "embedding_dim": embedding_model.get_sentence_embedding_dimension(),
        "purpose": "dense_retrieval",
    })

    mlflow.sentence_transformers.log_model(
        embedding_model, artifact_path="embedding_model",
        registered_model_name=EMBEDDING_REGISTRY, signature=signature,
    )

print(f"Registered: {EMBEDDING_REGISTRY}")
print(f"Embedding dim: {embedding_model.get_sentence_embedding_dimension()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Register Cross-Encoder Reranker

# COMMAND ----------

from sentence_transformers import CrossEncoder
import os, tempfile

RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANKER_REGISTRY = f"{CATALOG}.{SCHEMA}.reranker_cross_encoder"

reranker_model = CrossEncoder(RERANKER_MODEL_NAME)

# Save model locally so MLflow can log it as an artifact
reranker_dir = tempfile.mkdtemp()
reranker_model.save(reranker_dir)

# Create signature
sample_pairs = [["What causes diabetes?", "Diabetes is caused by insulin resistance."]]
sample_scores = reranker_model.predict(sample_pairs)
signature = infer_signature(
    pd.DataFrame({"query": ["What causes diabetes?"], "document": ["Diabetes is caused by insulin resistance."]}),
    sample_scores,
)

with mlflow.start_run(run_name="reranker_cross_encoder"):
    mlflow.log_params({
        "model_name": RERANKER_MODEL_NAME,
        "model_type": "CrossEncoder",
        "purpose": "two_stage_reranking",
    })

    # Log the saved model directory as artifacts + register as pyfunc
    class RerankerWrapper(mlflow.pyfunc.PythonModel):
        def load_context(self, context):
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(context.artifacts["reranker_dir"])

        def predict(self, context, model_input):
            pairs = list(zip(model_input["query"], model_input["document"]))
            return self.model.predict(pairs)

    mlflow.pyfunc.log_model(
        artifact_path="reranker_model",
        registered_model_name=RERANKER_REGISTRY,
        signature=signature,
        python_model=RerankerWrapper(),
        artifacts={"reranker_dir": reranker_dir},
        extra_pip_requirements=["sentence-transformers"],
    )

print(f"Registered: {RERANKER_REGISTRY}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Verify All Registered Models

# COMMAND ----------

from mlflow import MlflowClient

client = MlflowClient()

registered_models = [ATRISK_MODEL, EMBEDDING_REGISTRY, RERANKER_REGISTRY]

print("=== Unity Catalog Model Registry ===\n")
for model_name in registered_models:
    try:
        info = client.get_registered_model(model_name)
        latest = info.latest_versions[0] if info.latest_versions else None
        version = latest.version if latest else "none"
        print(f"  {model_name}")
        print(f"    Version: {version}")
        print()
    except Exception as e:
        print(f"  {model_name}: {e}\n")

print("All pipeline models registered.")
