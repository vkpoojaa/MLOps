#!/usr/bin/env python3
"""Small PR smoke run: configured ingestion -> validation -> Data Gate.

Intentionally does not train models, contact MLflow, register a model, or promote
production. CI success is not an authorization to bypass either ML quality gate.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlops_pipeline.config import load_config  # noqa: E402
from mlops_pipeline.pipeline.prefect_flow import (  # noqa: E402
    data_gate_task,
    flow,
    ingestion_task,
    validation_task,
)


@flow(name="ci-data-gate-smoke")
def run_smoke():
    settings = load_config(check_paths=True)
    frame = ingestion_task(settings)
    report = validation_task(frame, settings)
    _call = data_gate_task(frame, report)
    return len(frame), len(frame.columns), len(report.results), len(report.failures)


def main() -> int:
    try:
        rows, columns, checks, failures = run_smoke()
    except Exception as exc:
        print(f"CI PIPELINE SMOKE FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print("CI PIPELINE SMOKE PASSED")
    print(f"rows={rows} columns={columns}")
    print(f"data_gate=PASS checks={checks} failures={failures}")
    print("scope=ingestion+validation+Data Gate; no training/registry/promotion")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
