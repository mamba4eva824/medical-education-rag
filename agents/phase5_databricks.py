#!/usr/bin/env python3
"""
Phase 5 Agent: Databricks Porting
===================================
Job Responsibility: Build and maintain ML pipelines in Databricks +
Ensure performance, reliability, and scalability + Full lifecycle management

Validates:
- Export data ready for Databricks upload
- Databricks notebook scripts exist and are well-structured
- Local MLflow experiments exist to be referenced
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.base import PhaseAgent


class Phase5Agent(PhaseAgent):
    phase_name = "Phase 5: Databricks Porting"

    def run(self):
        # --- Export data ---
        self.check(
            "Export notebook exists",
            self.file_has_content, "notebooks/04_export_for_databricks.ipynb"
        )
        self.check(
            "Export CSV data exists",
            self._test_export_data
        )

        # --- Databricks notebooks ---
        db_files = {
            "Delta tables notebook": "databricks/db_01_delta_tables.py",
            "MLflow experiments notebook": "databricks/db_02_mlflow_experiments.py",
            "Model registry notebook": "databricks/db_03_model_registry.py",
        }
        for name, path in db_files.items():
            self.check(f"{name} exists", self.file_has_content, path)

        # --- Content validation ---
        self.check("Delta tables notebook uses spark.read", self._test_delta_content)
        self.check("MLflow notebook uses mlflow.start_run", self._test_mlflow_content)
        self.check("Registry notebook registers model", self._test_registry_content)

        # --- Prerequisites from earlier phases ---
        self.check("Local MLflow experiments exist", self._test_local_mlflow)
        self.check("All prior phases complete", self._test_prior_phases)

    def _test_export_data(self) -> tuple[bool, str]:
        exports_dir = self.project_root / "data" / "exports"
        csv_files = list(exports_dir.glob("*.csv"))
        if csv_files:
            total_size = sum(f.stat().st_size for f in csv_files)
            return True, f"Found {len(csv_files)} CSV(s), {total_size / 1024:.0f}KB total"
        return False, "No CSV exports — run notebook 04 first"

    def _test_delta_content(self) -> tuple[bool, str]:
        path = self.project_root / "databricks" / "db_01_delta_tables.py"
        if not path.exists():
            return False, "File missing"
        content = path.read_text()
        checks = {
            'spark.read': 'spark.read' in content,
            'delta': 'delta' in content.lower(),
            'saveAsTable': 'saveAsTable' in content or 'save_as_table' in content,
        }
        missing = [k for k, v in checks.items() if not v]
        if missing:
            return False, f"Missing patterns: {missing}"
        return True, "Contains spark.read, delta format, and saveAsTable"

    def _test_mlflow_content(self) -> tuple[bool, str]:
        path = self.project_root / "databricks" / "db_02_mlflow_experiments.py"
        if not path.exists():
            return False, "File missing"
        content = path.read_text()
        checks = {
            'mlflow.start_run': 'mlflow.start_run' in content or 'start_run' in content,
            'set_experiment': 'set_experiment' in content,
            'SentenceTransformer': 'SentenceTransformer' in content,
        }
        missing = [k for k, v in checks.items() if not v]
        if missing:
            return False, f"Missing patterns: {missing}"
        return True, "Contains MLflow experiment setup and model loading"

    def _test_registry_content(self) -> tuple[bool, str]:
        path = self.project_root / "databricks" / "db_03_model_registry.py"
        if not path.exists():
            return False, "File missing"
        content = path.read_text()
        if 'registered_model_name' not in content and 'register_model' not in content:
            return False, "Missing model registration call"
        return True, "Contains model registration"

    def _test_local_mlflow(self) -> tuple[bool, str]:
        mlruns = self.project_root / "mlruns"
        if not mlruns.exists():
            return False, "No mlruns/ directory — run experiments first"
        experiments = list(mlruns.glob("*/meta.yaml"))
        if len(experiments) < 2:
            return False, f"Only {len(experiments)} experiment(s) — need embedding + at_risk"
        return True, f"Found {len(experiments)} MLflow experiments"

    def _test_prior_phases(self) -> tuple[bool, str]:
        critical_files = [
            "src/ingestion/chunker.py",
            "src/embeddings/vector_store.py",
            "src/retrieval/reranker.py",
            "src/generation/guardrails.py",
            "src/prediction/at_risk_model.py",
            "src/api/main.py",
        ]
        missing = [f for f in critical_files
                   if not (self.project_root / f).exists()
                   or (self.project_root / f).stat().st_size < 100]
        if missing:
            return False, f"Missing/empty files from prior phases: {missing}"
        return True, "All prior phase deliverables present"


if __name__ == "__main__":
    agent = Phase5Agent()
    report = agent.execute()
    sys.exit(0 if report.all_passed else 1)
