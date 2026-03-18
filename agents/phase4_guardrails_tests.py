#!/usr/bin/env python3
"""
Phase 4 Agent: Guardrails, Prediction & Tests
===============================================
Job Responsibility: Ensure responsible AI practices + Predictive modeling
for learner outcomes + MLOps practices

Validates:
- Guardrails module (validation + content filtering)
- At-risk learner prediction model (classification)
- Retrieval quality predictor training (regression)
- Monitoring module
- Test suite completeness and passing
"""

import sys
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.base import PhaseAgent


class Phase4Agent(PhaseAgent):
    phase_name = "Phase 4: Guardrails, Prediction & Tests"

    def run(self):
        # --- File existence ---
        files = {
            "guardrails.py": "src/generation/guardrails.py",
            "at_risk_model.py": "src/prediction/at_risk_model.py",
            "monitoring.py": "src/api/monitoring.py",
            "test_retrieval.py": "tests/test_retrieval.py",
            "test_quality_predictor.py": "tests/test_quality_predictor.py",
            "test_guardrails.py": "tests/test_guardrails.py",
            "test_api.py": "tests/test_api.py",
            "Predictive model notebook": "notebooks/05_predictive_model.ipynb",
            "Quality predictor notebook": "notebooks/05b_quality_predictor.ipynb",
        }
        for name, path in files.items():
            self.check(f"{name} exists", self.file_has_content, path)

        # --- Import checks ---
        self.check(
            "guardrails imports cleanly",
            self.module_imports, "src.generation.guardrails"
        )
        self.check(
            "at_risk_model imports cleanly",
            self.module_imports, "src.prediction.at_risk_model"
        )
        self.check(
            "monitoring imports cleanly",
            self.module_imports, "src.api.monitoring"
        )

        # --- Functional checks ---
        self.check("Guardrails validate_response works", self._test_guardrails)
        self.check("AtRiskPipeline has required methods", self._test_at_risk_interface)
        self.check("QueryMetrics tracks correctly", self._test_monitoring)
        self.check("pytest suite passes", self._test_pytest)

    def _test_guardrails(self) -> tuple[bool, str]:
        try:
            from src.generation.guardrails import validate_response

            # Test that prohibited content is caught
            mock_sources = [
                {'doc': {'text': 'Heart failure symptoms include edema and fatigue.'}},
            ]
            bad_response = "You should self-diagnose this condition."
            result = validate_response(bad_response, mock_sources)

            if not isinstance(result, dict):
                return False, "validate_response should return a dict"
            if 'passed' not in result:
                return False, "Result missing 'passed' key"
            if 'within_scope' not in result:
                return False, "Result missing 'within_scope' key"
            if result.get('within_scope') is not False:
                return False, "Should catch prohibited phrase 'self-diagnose'"
            return True, "Guardrails correctly catch prohibited content"
        except Exception as e:
            return False, str(e)

    def _test_at_risk_interface(self) -> tuple[bool, str]:
        try:
            from src.prediction.at_risk_model import AtRiskPipeline
            required = ['create_features', 'train_and_log']
            missing = [m for m in required if not hasattr(AtRiskPipeline, m)]
            if missing:
                return False, f"Missing methods: {missing}"
            return True, "AtRiskPipeline has create_features() and train_and_log()"
        except Exception as e:
            return False, str(e)

    def _test_monitoring(self) -> tuple[bool, str]:
        try:
            from src.api.monitoring import QueryMetrics
            metrics = QueryMetrics()
            metrics.record(latency=150.0, passed=True, empty=False)
            metrics.record(latency=250.0, passed=False, empty=False)
            metrics.record(latency=100.0, passed=True, empty=True)
            summary = metrics.summary()

            if summary['total_queries'] != 3:
                return False, f"Expected 3 queries, got {summary['total_queries']}"
            if summary['guardrail_fail_rate'] == 0:
                return False, "Should have non-zero guardrail failure rate"
            if 'p50_latency_ms' not in summary:
                return False, "Missing p50_latency_ms in summary"
            return True, f"QueryMetrics working: {summary['total_queries']} queries tracked"
        except Exception as e:
            return False, str(e)

    def _test_pytest(self) -> tuple[bool, str]:
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
                capture_output=True, text=True, timeout=120,
                cwd=str(self.project_root)
            )
            # Count passed/failed from output
            output = result.stdout + result.stderr
            if result.returncode == 0:
                return True, "All tests passed"
            else:
                # Extract failure summary
                lines = output.strip().split('\n')
                summary = [l for l in lines if 'passed' in l or 'failed' in l]
                msg = summary[-1] if summary else f"Exit code {result.returncode}"
                return False, msg
        except subprocess.TimeoutExpired:
            return False, "Tests timed out after 120s"
        except Exception as e:
            return False, str(e)


if __name__ == "__main__":
    agent = Phase4Agent()
    report = agent.execute()
    sys.exit(0 if report.all_passed else 1)
