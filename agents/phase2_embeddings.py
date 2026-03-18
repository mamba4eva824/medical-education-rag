#!/usr/bin/env python3
"""
Phase 2 Agent: Embeddings, Vector Store & Recommendations
==========================================================
Job Responsibility: Evaluate vendor versus open-source AI products
based on performance, cost, and reliability

Validates:
- Vector store (ChromaDB wrapper)
- Content recommender
- Embedding comparison MLflow experiments
- ChromaDB persistence
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.base import PhaseAgent


class Phase2Agent(PhaseAgent):
    phase_name = "Phase 2: Embeddings, Vector Store & Recommendations"

    def run(self):
        # --- File existence ---
        self.check(
            "vector_store.py exists",
            self.file_has_content, "src/embeddings/vector_store.py"
        )
        self.check(
            "recommender.py exists",
            self.file_has_content, "src/embeddings/recommender.py"
        )
        self.check(
            "Embedding comparison notebook exists",
            self.file_has_content, "notebooks/02_embedding_comparison.ipynb"
        )

        # --- Import checks ---
        self.check(
            "vector_store imports cleanly",
            self.module_imports, "src.embeddings.vector_store"
        )
        self.check(
            "recommender imports cleanly",
            self.module_imports, "src.embeddings.recommender"
        )

        # --- Functional checks ---
        self.check("VectorStore has required methods", self._test_vector_store_interface)
        self.check("ContentRecommender has required methods", self._test_recommender_interface)
        self.check("ChromaDB persistence directory exists", self._test_chroma_persistence)
        self.check("MLflow runs exist for embedding comparison", self._test_mlflow_runs)

    def _test_vector_store_interface(self) -> tuple[bool, str]:
        try:
            from src.embeddings.vector_store import VectorStore
            required = ['build_index', 'search']
            missing = [m for m in required if not hasattr(VectorStore, m)]
            if missing:
                return False, f"Missing methods: {missing}"
            return True, "VectorStore has build_index() and search()"
        except Exception as e:
            return False, str(e)

    def _test_recommender_interface(self) -> tuple[bool, str]:
        try:
            from src.embeddings.recommender import ContentRecommender
            required = ['get_similar', 'recommend_study_path']
            missing = [m for m in required if not hasattr(ContentRecommender, m)]
            if missing:
                return False, f"Missing methods: {missing}"
            return True, "ContentRecommender has get_similar() and recommend_study_path()"
        except Exception as e:
            return False, str(e)

    def _test_chroma_persistence(self) -> tuple[bool, str]:
        chroma_dir = self.project_root / "chroma_db"
        if chroma_dir.exists() and any(chroma_dir.iterdir()):
            return True, f"ChromaDB data found at {chroma_dir}"
        return False, "No ChromaDB data — run notebook 02b to build the index"

    def _test_mlflow_runs(self) -> tuple[bool, str]:
        try:
            import mlflow
            mlflow.set_tracking_uri(str(self.project_root / "mlruns"))
            client = mlflow.tracking.MlflowClient()
            experiments = client.search_experiments()
            embedding_exps = [
                e for e in experiments
                if 'embedding' in (e.name or '').lower()
            ]
            if not embedding_exps:
                return False, "No embedding_comparison experiment found in MLflow"
            exp = embedding_exps[0]
            runs = client.search_runs(experiment_ids=[exp.experiment_id])
            if len(runs) >= 2:
                return True, f"Found {len(runs)} runs in '{exp.name}'"
            return False, f"Only {len(runs)} run(s) — need at least 2 models compared"
        except Exception as e:
            return False, str(e)


if __name__ == "__main__":
    agent = Phase2Agent()
    report = agent.execute()
    sys.exit(0 if report.all_passed else 1)
