"""Base agent class for development lifecycle phases."""

import sys
import time
import importlib
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class CheckResult:
    name: str
    passed: bool
    message: str
    duration_sec: float = 0.0


@dataclass
class PhaseReport:
    phase: str
    checks: list[CheckResult] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0

    @property
    def all_passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def passed_count(self) -> int:
        return sum(1 for c in self.checks if c.passed)

    @property
    def failed_count(self) -> int:
        return sum(1 for c in self.checks if not c.passed)

    def print_report(self):
        elapsed = self.end_time - self.start_time
        print(f"\n{'=' * 60}")
        print(f"  Phase: {self.phase}")
        print(f"  Duration: {elapsed:.1f}s")
        print(f"  Results: {self.passed_count} passed, {self.failed_count} failed")
        print(f"{'=' * 60}\n")

        for check in self.checks:
            icon = "PASS" if check.passed else "FAIL"
            print(f"  [{icon}] {check.name}")
            if check.message:
                print(f"         {check.message}")
            if check.duration_sec > 0:
                print(f"         ({check.duration_sec:.2f}s)")

        print()
        if self.all_passed:
            print("  >>> ALL CHECKS PASSED — Phase complete!")
        else:
            print("  >>> SOME CHECKS FAILED — Review above and fix before proceeding.")
        print()


class PhaseAgent:
    """Base class for lifecycle phase agents."""

    phase_name: str = "base"

    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.report = PhaseReport(phase=self.phase_name)

    def check(self, name: str, fn, *args, **kwargs) -> CheckResult:
        """Run a single check and record the result."""
        start = time.time()
        try:
            result = fn(*args, **kwargs)
            duration = time.time() - start
            if isinstance(result, tuple):
                passed, message = result
            elif isinstance(result, bool):
                passed, message = result, ""
            else:
                passed, message = bool(result), str(result)
            cr = CheckResult(name=name, passed=passed, message=message, duration_sec=duration)
        except Exception as e:
            duration = time.time() - start
            cr = CheckResult(name=name, passed=False, message=str(e), duration_sec=duration)
        self.report.checks.append(cr)
        return cr

    def file_exists(self, relative_path: str) -> tuple[bool, str]:
        """Check if a file exists relative to project root."""
        path = self.project_root / relative_path
        if path.exists():
            size = path.stat().st_size
            return True, f"Found ({size} bytes)"
        return False, f"Missing: {path}"

    def file_has_content(self, relative_path: str, min_bytes: int = 100) -> tuple[bool, str]:
        """Check if a file exists and has meaningful content."""
        path = self.project_root / relative_path
        if not path.exists():
            return False, f"Missing: {path}"
        size = path.stat().st_size
        if size < min_bytes:
            return False, f"Too small ({size} bytes, need {min_bytes}+)"
        return True, f"Found ({size} bytes)"

    def module_imports(self, module_path: str) -> tuple[bool, str]:
        """Check if a Python module can be imported."""
        try:
            importlib.import_module(module_path)
            return True, f"Successfully imported {module_path}"
        except ImportError as e:
            return False, f"Import failed: {e}"
        except Exception as e:
            return False, f"Error: {e}"

    def directory_has_files(self, relative_path: str, pattern: str = "*") -> tuple[bool, str]:
        """Check if a directory has files matching a pattern."""
        path = self.project_root / relative_path
        if not path.exists():
            return False, f"Directory missing: {path}"
        files = list(path.glob(pattern))
        if not files:
            return False, f"No files matching '{pattern}' in {path}"
        return True, f"Found {len(files)} file(s)"

    def run(self):
        """Override this in subclasses to define phase checks."""
        raise NotImplementedError

    def execute(self):
        """Run all checks and print the report."""
        self.report.start_time = time.time()
        try:
            self.run()
        except Exception as e:
            self.report.checks.append(
                CheckResult(name="Agent execution", passed=False, message=str(e))
            )
        self.report.end_time = time.time()
        self.report.print_report()
        return self.report
