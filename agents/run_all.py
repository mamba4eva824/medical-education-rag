#!/usr/bin/env python3
"""
Development Lifecycle Orchestrator
====================================
Runs all phase agents in sequence, or a specific phase.

Usage:
    python agents/run_all.py          # Run all phases, stop at first incomplete
    python agents/run_all.py 1        # Run phase 1 only
    python agents/run_all.py 3        # Run phase 3 only
    python agents/run_all.py --all    # Run all phases regardless of failures
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.phase1_ingestion import Phase1Agent
from agents.phase2_embeddings import Phase2Agent
from agents.phase3_rag_api import Phase3Agent
from agents.phase4_guardrails_tests import Phase4Agent
from agents.phase5_databricks import Phase5Agent

PHASES = {
    1: Phase1Agent,
    2: Phase2Agent,
    3: Phase3Agent,
    4: Phase4Agent,
    5: Phase5Agent,
}


def main():
    args = sys.argv[1:]

    if args and args[0] == "--all":
        run_phases = list(PHASES.keys())
        stop_on_fail = False
    elif args and args[0].isdigit():
        phase_num = int(args[0])
        if phase_num not in PHASES:
            print(f"Unknown phase {phase_num}. Valid: {list(PHASES.keys())}")
            sys.exit(1)
        run_phases = [phase_num]
        stop_on_fail = False
    else:
        run_phases = list(PHASES.keys())
        stop_on_fail = True

    print("=" * 60)
    print("  Medical Education RAG — Development Lifecycle Check")
    print("=" * 60)

    results = {}
    for phase_num in run_phases:
        agent_class = PHASES[phase_num]
        agent = agent_class()
        report = agent.execute()
        results[phase_num] = report

        if stop_on_fail and not report.all_passed:
            print(f"\n>>> Phase {phase_num} incomplete. Fix issues above, then re-run.")
            print(f">>> To run just this phase: python agents/run_all.py {phase_num}")
            break

    # Summary
    print("\n" + "=" * 60)
    print("  LIFECYCLE SUMMARY")
    print("=" * 60)
    for phase_num, report in results.items():
        status = "COMPLETE" if report.all_passed else "INCOMPLETE"
        print(f"  Phase {phase_num}: {report.phase} — [{status}] "
              f"({report.passed_count}/{report.passed_count + report.failed_count} checks)")
    print()

    all_done = all(r.all_passed for r in results.values())
    sys.exit(0 if all_done else 1)


if __name__ == "__main__":
    main()
