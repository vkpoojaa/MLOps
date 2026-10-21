"""Repeat the configured training procedure and compare metrics without side effects."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ..config.loader import Settings
from .mlflow_tracking import code_version, sha256_file


def check_reproducibility(settings: Settings) -> dict[str, Any]:
    """Run training twice with identical inputs and compare configured metrics.

    Uses the tested project training entry point with ``save=False`` so this
    check does not replace candidate or production artifacts or create MLflow runs.
    """
    from ..models.training import run_training

    data_hash = sha256_file(settings.train_path)
    config_hash = sha256_file(settings.config_path)
    version = code_version(settings.project_root)
    source_digest = hashlib.sha256()
    source_root = settings.project_root / "src" / "mlops_pipeline"
    for source_file in sorted(source_root.rglob("*.py")):
        source_digest.update(source_file.relative_to(settings.project_root).as_posix().encode("utf-8"))
        source_digest.update(source_file.read_bytes())
    source_tree_sha256 = source_digest.hexdigest()
    first = run_training(settings, save=False)
    second = run_training(settings, save=False)
    tolerance = settings.config.reproducibility.metric_tolerance
    comparisons: list[dict[str, Any]] = []
    for role in first.results:
        for metric in settings.config.metric_names:
            left = float(first.results[role].validation_metrics[metric])
            right = float(second.results[role].validation_metrics[metric])
            delta = abs(left - right)
            comparisons.append({"role": role, "metric": metric, "run_1": left,
                                "run_2": right, "absolute_delta": delta,
                                "tolerance": tolerance, "passed": delta <= tolerance})
    # These hashes are captured before the runs; also verify inputs did not change.
    input_unchanged = data_hash == sha256_file(settings.train_path) and config_hash == sha256_file(settings.config_path)
    passed = input_unchanged and all(item["passed"] for item in comparisons)
    report = {
        "status": "PASS" if passed else "FAIL",
        "method": "two independent run_training(save=False) executions in the same process/environment",
        "dataset_identifier": settings.train_path.name,
        "dataset_sha256": data_hash,
        "config_path": str(settings.config_path.relative_to(settings.project_root)),
        "config_sha256": config_hash,
        "code_version": version,
        "source_tree_sha256": source_tree_sha256,
        "random_seed": settings.config.random_seed,
        "metric_tolerance": tolerance,
        "tolerance_rationale": "Existing real-data reproducibility test asserts exact metric equality for repeated runs. The configured 1e-12 allows only floating-point comparison noise; it is not a model-quality margin.",
        "inputs_unchanged": input_unchanged,
        "comparisons": comparisons,
    }
    out = settings.resolve(settings.config.paths.reports_dir) / "reproducibility_check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report["report_path"] = str(out.relative_to(settings.project_root))
    return report
