"""MLflow experiment tracking helpers used by the pipeline itself.

Runs are opened before each model is fitted, not created manually afterward.
MLflow is required when ``mlflow.enabled`` is true; absence is a hard error.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from ..config.loader import Settings
from ..data.validation import ValidationReport


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def code_version(project_root: Path) -> str:
    """Return commit hash when available; never invent a commit identifier."""
    env_value = os.environ.get("GIT_COMMIT")
    if env_value:
        return env_value
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=project_root,
            check=True, capture_output=True, text=True, timeout=2,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable-no-git-commit"


def load_mlflow(settings: Settings):
    if not settings.config.mlflow.enabled:
        return None
    try:
        import mlflow
        import mlflow.sklearn
    except ImportError as exc:
        raise RuntimeError(
            "MLflow tracking is enabled in config.yaml but MLflow is not installed. "
            "Install project dependencies with `pip install -r requirements.txt`."
        ) from exc

    uri = settings.config.mlflow.tracking_uri
    if uri.startswith("sqlite:///"):
        db_part = uri.removeprefix("sqlite:///")
        if db_part and not Path(db_part).is_absolute():
            uri = "sqlite:///" + str((settings.project_root / db_part).resolve())
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(settings.config.mlflow.experiment_name)
    return mlflow


def _flatten_params(prefix: str, value: Any) -> dict[str, str]:
    output: dict[str, str] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            output.update(_flatten_params(f"{prefix}.{key}" if prefix else str(key), child))
    elif isinstance(value, (list, tuple)):
        output[prefix] = json.dumps(value, sort_keys=True, default=str)
    else:
        output[prefix] = str(value)
    return output


def fit_and_track_model(
    *, settings: Settings, role: str, spec, dataset_path: Path,
    validation_report: ValidationReport, fit_callable,
):
    """Create the MLflow run first, then fit and log the resulting model."""
    mlflow = load_mlflow(settings)
    if mlflow is None:
        return fit_callable(), None

    cfg = settings.config
    artifact_root = cfg.mlflow.artifact_subdirectory.strip("/")
    dataset_hash = sha256_file(dataset_path)
    config_hash = sha256_file(settings.config_path)
    version = code_version(settings.project_root)
    run_name = f"{role}-{spec.name}"
    with mlflow.start_run(run_name=run_name, description="Created by Project 4 pipeline") as active:
        mlflow.set_tags({
            "project": cfg.project.name,
            "model_role": role,
            "dataset.identifier": dataset_path.name,
            "dataset.sha256": dataset_hash,
            "config.path": str(settings.config_path.relative_to(settings.project_root)),
            "config.sha256": config_hash,
            "code.version": version,
            "pipeline.stage": "training-started",
            "validation.passed": str(validation_report.passed).lower(),
        })
        params = {
            "model.type": spec.algorithm,
            "model.name": spec.name,
            "model.role": role,
            "model.hyperparameters": spec.hyperparameters,
            "preprocessing": cfg.preprocessing.model_dump(mode="json"),
            "split.test_size": cfg.model.split.test_size,
            "evaluation.primary_metric": cfg.model.evaluation.primary_metric,
            "evaluation.cv_folds": cfg.model.evaluation.cv_folds,
            "random_seed": cfg.random_seed,
            "dataset.identifier": dataset_path.name,
            "dataset.sha256": dataset_hash,
            "config.sha256": config_hash,
            "code.version": version,
        }
        for key, value in _flatten_params("", params).items():
            mlflow.log_param(key, value[:500])
        mlflow.log_artifact(str(settings.config_path), artifact_path=f"{artifact_root}/configuration")
        with tempfile.TemporaryDirectory(prefix="phase5-validation-") as tmp:
            validation_path = Path(tmp) / "validation_results.json"
            validation_path.write_text(json.dumps(validation_report.to_dict(), indent=2, default=str), encoding="utf-8")
            mlflow.log_artifact(str(validation_path), artifact_path=f"{artifact_root}/validation")

        # The run exists before any model fit begins. An exception marks it FAILED.
        result = fit_callable()
        mlflow.log_metrics({key.upper(): float(value) for key, value in result.validation_metrics.items()})
        mlflow.sklearn.log_model(result.pipeline, artifact_path=f"{artifact_root}/model")
        mlflow.set_tag("pipeline.stage", "training-complete")
        run_id = active.info.run_id
        result.extras["mlflow_run_id"] = run_id
        result.extras["dataset_identifier"] = str(dataset_path.relative_to(settings.project_root)) if dataset_path.is_relative_to(settings.project_root) else dataset_path.name
        result.extras["dataset_sha256"] = dataset_hash
        result.extras["config_path"] = str(settings.config_path.relative_to(settings.project_root))
        result.extras["config_sha256"] = config_hash
        result.extras["code_version"] = version
    return result, run_id


def log_evaluation_artifact(settings: Settings, run_id: str | None, report_path: Path) -> None:
    """Attach final comparison/evaluation evidence to the already-created run."""
    if not run_id or not settings.config.mlflow.enabled:
        return
    mlflow = load_mlflow(settings)
    assert mlflow is not None
    artifact_root = settings.config.mlflow.artifact_subdirectory.strip("/")
    with mlflow.start_run(run_id=run_id):
        mlflow.log_artifact(str(report_path), artifact_path=f"{artifact_root}/evaluation")
        mlflow.set_tag("pipeline.stage", "evaluation-complete")
