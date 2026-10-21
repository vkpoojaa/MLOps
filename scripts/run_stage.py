#!/usr/bin/env python3
"""Run one pipeline stage independently for audit/reproduction purposes.

Stages reuse the production Prefect task implementations; no alternative ML logic
is introduced. Downstream stages consume a persisted JSON/CSV/joblib artifact or
are explicitly documented as requiring the output of a prior stage.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mlops_pipeline.config import load_config
from mlops_pipeline.pipeline.prefect_flow import (
    data_gate_task,
    evaluation_task,
    feature_engineering_task,
    ingestion_task,
    model_quality_gate_task,
    registry_promotion_task,
    training_task,
    validation_task,
)
from mlops_pipeline.monitoring.drift import monitor_incoming_batches
from mlops_pipeline.models.quality_gate import load_production_record
from mlops_pipeline.models.training import ROLES


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["ingest", "validate", "data-gate", "feature-engineering", "train", "evaluate", "model-gate", "monitor"])
    p.add_argument("--config", default=None)
    p.add_argument("--data-path", default=None)
    p.add_argument("--output-dir", default=None)
    args = p.parse_args()
    settings = load_config(args.config, check_paths=True)
    frame = ingestion_task(settings, args.data_path)
    if args.stage == "ingest":
        print(json.dumps({"rows": len(frame), "columns": list(frame.columns)}, indent=2))
        return 0
    report = validation_task(frame, settings)
    if args.stage == "validate":
        print(report.to_text())
        return 0 if report.passed else 1
    gated = data_gate_task(frame, report)
    if args.stage == "data-gate":
        print(json.dumps({"status": "PASS", "rows": len(gated)}, indent=2))
        return 0
    split = feature_engineering_task(gated, settings)
    if args.stage == "feature-engineering":
        print(json.dumps({"train_rows": split.sizes["train"], "validation_rows": split.sizes["validation"], "features": list(split.X_train.columns)}, indent=2))
        return 0
    dataset_path = Path(args.data_path).expanduser().resolve() if args.data_path else settings.train_path
    results = training_task(split, settings, report, dataset_path)
    if args.stage == "train":
        print(json.dumps({role: results[role].validation_metrics for role in ROLES}, indent=2))
        return 0
    evaluation = evaluation_task(gated, split, results, settings, dataset_path)
    if args.stage == "evaluate":
        print(json.dumps(evaluation, indent=2))
        return 0
    production_record = load_production_record(settings)
    gate = model_quality_gate_task(results, settings, production_record, evaluation["report_path"])
    if args.stage == "model-gate":
        print(json.dumps(gate, indent=2))
        return 0 if gate["passed"] else 1
    if args.stage == "monitor":
        result = monitor_incoming_batches(settings, output_dir=args.output_dir)
        print(json.dumps(result.report, indent=2))
        return 0
    raise AssertionError(args.stage)


if __name__ == "__main__":
    raise SystemExit(main())
