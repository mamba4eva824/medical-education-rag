"""Embedding model comparison: evaluate 3 models on retrieval quality using local cosine similarity.

Logs precision@5, MRR, and encoding time to MLflow for each model.
Ground truth: a retrieved chunk is relevant if its text has >50% token overlap
with the eval answer.
"""

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import mlflow
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


from src.utils.text import STOP_WORDS, token_overlap  # noqa: F401

MODELS = {
    "all-MiniLM-L6-v2": {"type": "open-source", "size": "80MB"},
    "all-mpnet-base-v2": {"type": "open-source", "size": "420MB"},
    "pritamdeka/S-PubMedBert-MS-MARCO": {"type": "domain-specific", "size": "420MB"},
}


def evaluate_model(
    model_name: str,
    chunk_texts: list[str],
    chunk_embeddings: np.ndarray,
    eval_questions: list[str],
    eval_answers: list[str],
    top_k: int = 5,
    relevance_threshold: float = 0.5,
) -> tuple[float, float]:
    """Evaluate retrieval using local cosine similarity.

    Returns (precision_at_k, mrr).
    A chunk is relevant if token overlap with the eval answer > threshold.
    """
    model = SentenceTransformer(model_name)
    query_embeddings = model.encode(eval_questions, show_progress_bar=False)

    # Normalize for cosine similarity
    chunk_norms = chunk_embeddings / np.linalg.norm(
        chunk_embeddings, axis=1, keepdims=True
    )
    query_norms = query_embeddings / np.linalg.norm(
        query_embeddings, axis=1, keepdims=True
    )

    # Batch cosine similarity: (n_queries, n_chunks)
    similarities = query_norms @ chunk_norms.T

    precisions = []
    reciprocal_ranks = []

    for i in range(len(eval_questions)):
        top_indices = np.argsort(similarities[i])[::-1][:top_k]
        retrieved_texts = [chunk_texts[idx] for idx in top_indices]

        # Check relevance via token overlap with eval answer
        relevant_count = 0
        first_relevant_rank = None
        for rank, text in enumerate(retrieved_texts):
            if token_overlap(eval_answers[i], text) > relevance_threshold:
                relevant_count += 1
                if first_relevant_rank is None:
                    first_relevant_rank = rank + 1

        precisions.append(relevant_count / top_k)
        reciprocal_ranks.append(
            1.0 / first_relevant_rank if first_relevant_rank else 0.0
        )

    return float(np.mean(precisions)), float(np.mean(reciprocal_ranks))


def main() -> None:
    mlflow.set_tracking_uri(str(PROJECT_ROOT / "mlruns"))
    mlflow.set_experiment("embedding_comparison")

    # Load data
    chunks_df = pd.read_parquet(PROJECT_ROOT / "data/processed/medical_chunks.parquet")
    eval_df = pd.read_parquet(PROJECT_ROOT / "data/processed/eval_queries.parquet")

    chunk_texts = chunks_df["text"].tolist()
    eval_questions = eval_df["question"].tolist()
    eval_answers = eval_df["answer"].tolist()

    print(f"Chunks: {len(chunk_texts)}, Eval queries: {len(eval_questions)}")

    for model_name, meta in MODELS.items():
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*60}")

        with mlflow.start_run(run_name=model_name):
            # Encode all chunks
            model = SentenceTransformer(model_name)
            start = time.time()
            chunk_embeddings = model.encode(
                chunk_texts, show_progress_bar=True, batch_size=256
            )
            encoding_time = time.time() - start
            print(f"Encoding time: {encoding_time:.1f}s")

            # Evaluate retrieval
            p5, mrr = evaluate_model(
                model_name, chunk_texts, chunk_embeddings,
                eval_questions, eval_answers,
            )
            print(f"Precision@5: {p5:.4f}")
            print(f"MRR: {mrr:.4f}")

            # Log to MLflow
            mlflow.log_params({
                **meta,
                "model": model_name,
                "embedding_dim": chunk_embeddings.shape[1],
            })
            mlflow.log_metrics({
                "precision_at_5": p5,
                "mrr": mrr,
                "encoding_time_sec": encoding_time,
            })

    print("\n\nDone. View results: mlflow ui --port 5000")


if __name__ == "__main__":
    main()
