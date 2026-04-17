"""A/B evaluation harness for comparing RAG pipeline strategies.

Runs both pipeline modes over eval_queries.parquet, computes retrieval
and answer quality metrics, and logs results to MLflow.
"""

import logging
import time
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from src.generation.rag_chain import RAGPipeline
from src.utils.text import token_overlap

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RELEVANCE_THRESHOLD = 0.3


class PipelineEvaluator:
    """Evaluates a RAG pipeline on held-out Q&A pairs."""

    def __init__(
        self,
        pipeline: RAGPipeline,
        eval_path: str | Path = "data/processed/eval_queries.parquet",
        mlflow_uri: str | None = None,
    ):
        self.pipeline = pipeline
        self.eval_df = pd.read_parquet(PROJECT_ROOT / eval_path)
        self.mlflow_uri = mlflow_uri or str(PROJECT_ROOT / "mlruns")

    def evaluate(
        self,
        n_queries: int | None = None,
        top_k: int = 20,
        rerank_n: int = 5,
        experiment_name: str = "pipeline_ab_comparison",
    ) -> dict:
        """Run pipeline over eval queries and compute metrics.

        Args:
            n_queries: Number of eval queries to use. None = all.
            top_k: Retrieval depth.
            rerank_n: Number of docs after reranking.
            experiment_name: MLflow experiment name.

        Returns:
            Dict with aggregate metrics and per-query results DataFrame.
        """
        eval_subset = self.eval_df.head(n_queries) if n_queries else self.eval_df
        mode = self.pipeline.strategy.name

        logger.info(
            f"Evaluating '{mode}' pipeline on {len(eval_subset)} queries..."
        )

        per_query_results = []

        for idx, row in eval_subset.iterrows():
            question = row["question"]
            ground_truth = row["answer"]
            qtype = row["qtype"]

            try:
                t0 = time.perf_counter()
                result = self.pipeline.answer(
                    query=question, top_k=top_k, rerank_n=rerank_n,
                )
                total_latency = time.perf_counter() - t0

                # Retrieval quality: check if reranked sources contain the answer
                sources = result["sources"]
                relevances = [
                    token_overlap(ground_truth, s.get("text", ""))
                    > RELEVANCE_THRESHOLD
                    for s in sources
                ]
                precision_at_k = sum(relevances) / len(relevances) if relevances else 0.0

                # MRR: reciprocal rank of first relevant result
                first_relevant = next(
                    (i + 1 for i, r in enumerate(relevances) if r), None,
                )
                mrr = 1.0 / first_relevant if first_relevant else 0.0

                # Answer quality: overlap with ground truth
                answer_overlap = token_overlap(ground_truth, result["answer"])

                # Guardrail pass
                passed = result["validation"].get("passed", False)

                per_query_results.append({
                    "question": question,
                    "qtype": qtype,
                    "precision_at_k": precision_at_k,
                    "mrr": mrr,
                    "answer_token_overlap": answer_overlap,
                    "guardrail_passed": passed,
                    "total_latency_s": total_latency,
                    **{f"timing_{k}": v for k, v in result["timing"].items()},
                    "llm_calls": result["api_calls"].get("llm_calls", 0),
                    "pinecone_calls": result["api_calls"].get("pinecone_calls", 0),
                    "pipeline_mode": mode,
                })

            except Exception as e:
                logger.warning(f"Query {idx} failed: {e}")
                per_query_results.append({
                    "question": question,
                    "qtype": qtype,
                    "precision_at_k": 0.0,
                    "mrr": 0.0,
                    "answer_token_overlap": 0.0,
                    "guardrail_passed": False,
                    "total_latency_s": 0.0,
                    "pipeline_mode": mode,
                    "error": str(e),
                })

        results_df = pd.DataFrame(per_query_results)

        # Aggregate metrics
        metrics = {
            "precision_at_5": float(results_df["precision_at_k"].mean()),
            "mrr": float(results_df["mrr"].mean()),
            "answer_token_overlap": float(results_df["answer_token_overlap"].mean()),
            "guardrail_pass_rate": float(results_df["guardrail_passed"].mean()),
            "mean_latency_s": float(results_df["total_latency_s"].mean()),
            "p95_latency_s": float(results_df["total_latency_s"].quantile(0.95)),
            "total_llm_calls": int(results_df.get("llm_calls", pd.Series([0])).sum()),
            "total_pinecone_calls": int(results_df.get("pinecone_calls", pd.Series([0])).sum()),
            "n_queries": len(results_df),
            "error_count": int(results_df.get("error", pd.Series()).notna().sum()),
        }

        # Log to MLflow
        mlflow.set_tracking_uri(self.mlflow_uri)
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run(run_name=f"pipeline_{mode}"):
            mlflow.log_params({
                "mode": mode,
                "n_queries": metrics["n_queries"],
                "top_k": top_k,
                "rerank_n": rerank_n,
            })
            mlflow.log_metrics(metrics)

            # Save per-query results as artifact
            artifact_path = PROJECT_ROOT / f"data/exports/eval_{mode}_results.csv"
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            results_df.to_csv(artifact_path, index=False)
            mlflow.log_artifact(str(artifact_path))

        logger.info(f"'{mode}' evaluation complete: {metrics}")

        return {
            "metrics": metrics,
            "per_query": results_df,
        }


def compare_pipelines(
    pipelines: dict[str, RAGPipeline],
    n_queries: int | None = None,
    experiment_name: str = "pipeline_ab_comparison",
) -> pd.DataFrame:
    """Run evaluation for multiple pipelines and return comparison table.

    Args:
        pipelines: Dict mapping mode name to RAGPipeline instance.
        n_queries: Number of eval queries. None = all.
        experiment_name: MLflow experiment name.

    Returns:
        DataFrame with one row per pipeline mode and metric columns.
    """
    all_metrics = []

    for mode_name, pipeline in pipelines.items():
        evaluator = PipelineEvaluator(pipeline=pipeline)
        result = evaluator.evaluate(
            n_queries=n_queries, experiment_name=experiment_name,
        )
        row = {"mode": mode_name, **result["metrics"]}
        all_metrics.append(row)

    comparison = pd.DataFrame(all_metrics).set_index("mode")
    return comparison
