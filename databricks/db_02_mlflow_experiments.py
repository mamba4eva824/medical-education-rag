# Databricks notebook source
# MAGIC %md
# MAGIC # 02 — Embedding Model Comparison with Databricks MLflow
# MAGIC
# MAGIC Re-runs the embedding model comparison from Phase 2, tracking experiments
# MAGIC in Databricks-managed MLflow instead of local file-based tracking.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Install Dependencies

# COMMAND ----------

# MAGIC %pip install sentence-transformers
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Configuration

# COMMAND ----------

import mlflow
import numpy as np
import time
from sentence_transformers import SentenceTransformer

CATALOG = "medical_education_rag_dbx"
SCHEMA = "rag_data"

# Set MLflow experiment in Databricks
mlflow.set_experiment(f"/Users/chris_weinreich@yahoo.com/embedding_comparison")

MODELS = {
    "all-MiniLM-L6-v2": {"type": "open-source", "size": "80MB"},
    "all-mpnet-base-v2": {"type": "open-source", "size": "420MB"},
    "pritamdeka/S-PubMedBert-MS-MARCO": {"type": "domain-specific", "size": "420MB"},
}

STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "dare", "ought",
    "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further", "then",
    "once", "and", "but", "or", "nor", "not", "so", "yet", "both",
    "either", "neither", "each", "every", "all", "any", "few", "more",
    "most", "other", "some", "such", "no", "only", "own", "same", "than",
    "too", "very", "just", "because", "if", "when", "while", "that",
    "this", "these", "those", "it", "its", "i", "me", "my", "we", "our",
    "you", "your", "he", "him", "his", "she", "her", "they", "them",
    "their", "what", "which", "who", "whom", "how", "where", "there",
}

def token_overlap(text_a, text_b):
    tokens_a = set(text_a.lower().split()) - STOP_WORDS
    tokens_b = set(text_b.lower().split()) - STOP_WORDS
    if not tokens_a:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a)

print("Configuration complete")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Load Data from Delta Tables

# COMMAND ----------

chunks_sdf = spark.table(f"{CATALOG}.{SCHEMA}.medical_chunks")
eval_sdf = spark.table(f"{CATALOG}.{SCHEMA}.eval_queries")

# Convert to pandas for embedding operations
chunks_pdf = chunks_sdf.toPandas()
eval_pdf = eval_sdf.toPandas()

# Filter out any None/NaN values from CSV round-trip
chunks_pdf = chunks_pdf.dropna(subset=["text"])
eval_pdf = eval_pdf.dropna(subset=["question", "answer"])

chunk_texts = chunks_pdf["text"].tolist()
eval_questions = eval_pdf["question"].tolist()[:100]  # Sample for speed
eval_answers = eval_pdf["answer"].tolist()[:100]

print(f"Chunks: {len(chunk_texts)}, Eval queries: {len(eval_questions)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Evaluate Each Embedding Model

# COMMAND ----------

for model_name, meta in MODELS.items():
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*60}")

    with mlflow.start_run(run_name=model_name):
        # Encode chunks
        model = SentenceTransformer(model_name)
        start = time.time()
        chunk_embeddings = model.encode(chunk_texts, show_progress_bar=True, batch_size=256)
        encoding_time = time.time() - start

        # Encode queries
        query_embeddings = model.encode(eval_questions, show_progress_bar=False)

        # Normalize for cosine similarity
        chunk_norms = chunk_embeddings / np.linalg.norm(chunk_embeddings, axis=1, keepdims=True)
        query_norms = query_embeddings / np.linalg.norm(query_embeddings, axis=1, keepdims=True)

        # Compute similarity and evaluate
        similarities = query_norms @ chunk_norms.T
        top_k = 5

        precisions = []
        reciprocal_ranks = []

        for i in range(len(eval_questions)):
            top_indices = np.argsort(similarities[i])[::-1][:top_k]
            retrieved_texts = [chunk_texts[idx] for idx in top_indices]

            relevant_count = 0
            first_relevant_rank = None
            for rank, text in enumerate(retrieved_texts):
                if token_overlap(eval_answers[i], text) > 0.5:
                    relevant_count += 1
                    if first_relevant_rank is None:
                        first_relevant_rank = rank + 1

            precisions.append(relevant_count / top_k)
            reciprocal_ranks.append(1.0 / first_relevant_rank if first_relevant_rank else 0.0)

        p5 = float(np.mean(precisions))
        mrr = float(np.mean(reciprocal_ranks))

        print(f"Precision@5: {p5:.4f}")
        print(f"MRR: {mrr:.4f}")
        print(f"Encoding time: {encoding_time:.1f}s")

        # Log to Databricks MLflow
        mlflow.log_params({**meta, "model": model_name, "embedding_dim": chunk_embeddings.shape[1]})
        mlflow.log_metrics({"precision_at_5": p5, "mrr": mrr, "encoding_time_sec": encoding_time})

print("\nDone. View results in MLflow Experiments sidebar.")
