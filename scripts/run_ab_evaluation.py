"""CLI entry point for A/B pipeline comparison.

Usage:
    python scripts/run_ab_evaluation.py              # Full 500 queries
    python scripts/run_ab_evaluation.py --n-queries 50  # Quick preview
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.embeddings.vector_store import VectorStore
from src.evaluation.eval_harness import compare_pipelines
from src.generation.llm_client import LLMClient
from src.generation.rag_chain import RAGPipeline
from src.retrieval.hybrid_search import HybridSearcher
from src.retrieval.reranker import Reranker
from src.retrieval.strategies import (
    DenseRetrievalStrategy,
    HybridRetrievalStrategy,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="A/B pipeline comparison")
    parser.add_argument(
        "--n-queries", type=int, default=None,
        help="Number of eval queries (default: all 500)",
    )
    args = parser.parse_args()

    print("Initializing components...")
    store = VectorStore()
    searcher = HybridSearcher(vector_store=store)
    reranker = Reranker()
    llm = LLMClient()

    # Share parquet data between strategies
    chunk_lookup = {m["chunk_id"]: m for m in searcher.chunk_metadata}

    hybrid_strategy = HybridRetrievalStrategy(searcher=searcher, llm_client=llm)
    dense_strategy = DenseRetrievalStrategy(
        vector_store=store, chunk_lookup=chunk_lookup,
    )

    pipelines = {
        "full": RAGPipeline(
            strategy=hybrid_strategy, reranker=reranker, llm_client=llm,
        ),
        "simple": RAGPipeline(
            strategy=dense_strategy, reranker=reranker, llm_client=llm,
        ),
    }

    n = args.n_queries
    print(f"Running A/B comparison on {n or 'all'} eval queries...\n")

    comparison = compare_pipelines(pipelines, n_queries=n)

    print("\n" + "=" * 70)
    print("A/B PIPELINE COMPARISON RESULTS")
    print("=" * 70)
    print(comparison.to_string())
    print("\nResults logged to MLflow. View: mlflow ui --port 5000")
    print(f"Per-query CSVs saved to data/exports/")


if __name__ == "__main__":
    main()
