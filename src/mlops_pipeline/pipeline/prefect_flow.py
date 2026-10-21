"""Prefect orchestration for the existing house-price ML pipeline.

The existing modelling/preprocessing algorithms are intentionally reused. Each
stage is a Prefect task so task states, logs, inputs, outputs, and failures are
visible in a Prefect flow run.
"""
from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
try:
    from prefect import flow, task
    from prefect.logging import get_run_logger
    PREFECT_AVAILABLE = True
except ImportError:  # permits local validation/tests before optional deps are installed
    import logging
    from functools import wraps

    PREFECT_AVAILABLE = False
    def get_run_logger():
        return logging.getLogger("mlops_pipeline.prefect_fallback")

    def _local_decorator(**_metadata):
        def decorate(func):
            @wraps(func)
            def wrapped(*args, **kwargs):
                return func(*args, **kwargs)
            wrapped.fn = func
            wrapped.prefect_metadata = dict(_metadata)
            return wrapped
        return decorate

    task = _local_decorator
    flow = _local_decorator

from ..config import Settings, load_config
from ..data import DataValidationError, DataValidator, ValidationReport, split_train_validation
from ..models.evaluation import HIGHER_IS_BETTER, ModelResult, best_by_primary_metric, evaluate_model, is_better
from ..models.leakage import check_leakage
from ..models.training import ROLES, _result_dict
from ..monitoring.mlflow_tracking import fit_and_track_model, log_evaluation_artifact
from ..monitoring.model_registry import register_and_promote_approved_candidate, rollback_production_to_version
from ..models.quality_gate import evaluate_model_gate, load_production_record, write_model_gate_report


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@task(name="1-ingestion", description="Load the configured historical training CSV", retries=0)
def ingestion_task(settings: Settings, data_path: str | None = None) -> pd.DataFrame:
    logger = get_run_logger()
    source = Path(data_path).expanduser().resolve() if data_path else settings.train_path
    if not source.is_file():
        raise FileNotFoundError(f"Training input does not exist: {source}")
    frame = pd.read_csv(source)
    logger.info("Ingested %s rows x %s columns from %s", len(frame), len(frame.columns), source)
    return frame


@task(name="2-validation", description="Run the configured Phase 3 data validator", retries=0)
def validation_task(frame: pd.DataFrame, settings: Settings) -> ValidationReport:
    logger = get_run_logger()
    report = DataValidator.from_settings(settings).validate(frame, "train")
    logger.info("Validation completed: passed=%s, checks=%s, failures=%s", report.passed,
                len(report.results), len(report.failures))
    if report.failures:
        for failure in report.failures:
            logger.error("Validation failure: %s column=%s observed=%s expected=%s reason=%s",
                         failure.name, failure.column, failure.observed, failure.expected, failure.reason)
    return report


@task(name="3-data-gate", description="Block all downstream tasks when validation fails", retries=0)
def data_gate_task(frame: pd.DataFrame, report: ValidationReport) -> pd.DataFrame:
    logger = get_run_logger()
    try:
        report.raise_if_failed()
    except DataValidationError:
        logger.error("DATA GATE BLOCKED: downstream feature engineering and training will not run")
        raise
    logger.info("DATA GATE PASSED: downstream tasks may proceed")
    return frame


@task(name="4-feature-engineering", description="Create the configured train/validation feature split", retries=0)
def feature_engineering_task(frame: pd.DataFrame, settings: Settings):
    logger = get_run_logger()
    split = split_train_validation(frame, settings.config)
    logger.info("Feature split created: train=%s rows, validation=%s rows, features=%s",
                split.sizes["train"], split.sizes["validation"], len(split.X_train.columns))
    return split


@task(name="5-training", description="Fit configured baseline and candidate models", retries=0)
def training_task(split, settings: Settings, validation_report: ValidationReport, dataset_path: Path) -> dict[str, ModelResult]:
    logger = get_run_logger()
    results: dict[str, ModelResult] = {}
    for role in ROLES:
        spec = settings.config.model_spec(role)
        logger.info("Creating MLflow run before fitting model role=%s type=%s", role, spec.algorithm)
        result, run_id = fit_and_track_model(
            settings=settings, role=role, spec=spec, dataset_path=dataset_path,
            validation_report=validation_report,
            fit_callable=lambda role=role, spec=spec: evaluate_model(role, spec, split, settings.config),
        )
        results[role] = result
        if run_id:
            logger.info("Tracked model role=%s with MLflow run_id=%s", role, run_id)
    logger.info("Training completed for %s models: %s", len(results), ", ".join(results))
    return results


def _comparison(results: dict[str, ModelResult], cfg) -> dict[str, Any]:
    base, candidate = results["baseline"], results["candidate"]
    primary = cfg.model.evaluation.primary_metric
    delta = {}
    for metric in cfg.metric_names:
        b, c = base.validation_metrics[metric], candidate.validation_metrics[metric]
        delta[metric] = {
            "baseline": b,
            "candidate": c,
            "candidate_minus_baseline": c - b,
            "candidate_relative_change": (c - b) / abs(b) if b else None,
            "candidate_better": is_better(metric, c, b),
        }
    f_base = np.array(base.cv_metrics[primary]["per_fold"])
    f_candidate = np.array(candidate.cv_metrics[primary]["per_fold"])
    diff = f_candidate - f_base
    wins = int((diff < 0).sum() if not HIGHER_IS_BETTER[primary] else (diff > 0).sum())
    return {
        "primary_metric": primary,
        "best_on_validation": best_by_primary_metric([base, candidate], cfg).role,
        "validation": delta,
        "cv_primary_metric_mean_std": {
            role: {key: results[role].cv_metrics[primary][key] for key in ("mean", "std")}
            for role in ROLES
        },
        "cv_primary_metric_paired_difference": {
            "candidate_minus_baseline_mean": float(diff.mean()),
            "candidate_minus_baseline_std": float(diff.std(ddof=1)),
            "folds_where_candidate_better": wins,
            "n_folds": int(len(diff)),
        },
    }


@task(name="6-evaluation", description="Evaluate checks, select configured model, and persist reports/artifacts", retries=0)
def evaluation_task(frame: pd.DataFrame, split, results: dict[str, ModelResult], settings: Settings, dataset_path: Path) -> dict[str, Any]:
    logger = get_run_logger()
    cfg = settings.config
    leakage = check_leakage(cfg, split, frame, list(results.values()))
    comparison = _comparison(results, cfg)
    selected_role = cfg.model.selected
    selection = {
        "selected_in_config": selected_role,
        "matches_best_on_validation": None if selected_role is None else selected_role == comparison["best_on_validation"],
    }
    saved: dict[str, str] = {}
    models_dir = settings.resolve(cfg.paths.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    for role, result in results.items():
        path = models_dir / f"{result.spec.name}.joblib"
        joblib.dump(result.pipeline, path)
        saved[role] = str(path.relative_to(settings.project_root))
    # Phase 6: evaluation may write candidate artifacts, but never replaces the
    # current production model. Promotion is explicitly out of scope.
    selection["selected_model_file"] = None
    selection["production_model_replaced"] = False
    report = {
        "settings": {
            "random_seed": cfg.random_seed,
            "test_size": cfg.model.split.test_size,
            "cv_folds": cfg.model.evaluation.cv_folds,
            "primary_metric": cfg.model.evaluation.primary_metric,
            "metrics": cfg.metric_names,
            "feature_columns": cfg.feature_columns,
            "excluded_columns": cfg.excluded_columns,
        },
        "data": {
            "training_file": cfg.paths.train_data,
            "training_file_sha256": _sha256(dataset_path),
            "dataset_identifier": dataset_path.name,
            "n_rows": len(frame),
            **{f"n_{key}": value for key, value in split.sizes.items()},
        },
        "models": {role: _result_dict(result) for role, result in results.items()},
        "comparison": comparison,
        "selection": selection,
        "leakage_checks": [check.to_dict() for check in leakage],
        "environment": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "orchestration": {"engine": "Prefect", "flow": "house-price-ml-pipeline"},
        "mlflow": {
            "enabled": cfg.mlflow.enabled,
            "experiment_name": cfg.mlflow.experiment_name,
            "tracking_uri": cfg.mlflow.tracking_uri,
            "runs": {
                role: {
                    "run_id": results[role].extras.get("mlflow_run_id"),
                    "dataset_sha256": results[role].extras.get("dataset_sha256"),
                    "config_sha256": results[role].extras.get("config_sha256"),
                    "code_version": results[role].extras.get("code_version"),
                }
                for role in ROLES
            },
        },
    }
    report_path = settings.resolve(cfg.paths.evaluation_report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    saved["report"] = str(report_path.relative_to(settings.project_root))
    for role, result in results.items():
        log_evaluation_artifact(settings, result.extras.get("mlflow_run_id"), report_path)
    summary = {
        "status": "completed",
        "rows": len(frame),
        "split_sizes": split.sizes,
        "primary_metric": cfg.model.evaluation.primary_metric,
        "best_on_validation": comparison["best_on_validation"],
        "selected_model": selected_role,
        "validation_metrics": {role: results[role].validation_metrics for role in ROLES},
        "mlflow_runs": {role: results[role].extras.get("mlflow_run_id") for role in ROLES},
        "leakage_checks_passed": sum(check.passed for check in leakage),
        "leakage_checks_total": len(leakage),
        "leakage_failures": [check.to_dict() for check in leakage if not check.passed],
        "saved_outputs": saved,
        "report_path": str(report_path.relative_to(settings.project_root)),
    }
    logger.info("Evaluation persisted to %s; artifacts=%s", summary["report_path"], saved)
    logger.info("Evaluation summary: primary_metric=%s best=%s selected=%s",
                summary["primary_metric"], summary["best_on_validation"], selected_role)
    return summary


@task(name="7-model-quality-gate", description="Apply absolute quality and current-production checks", retries=0)
def model_quality_gate_task(results: dict[str, ModelResult], settings: Settings,
                            production_record: dict[str, Any] | None,
                            evaluation_report_path: str) -> dict[str, Any]:
    logger = get_run_logger()
    candidate = results["candidate"]
    decision = evaluate_model_gate(candidate.validation_metrics, settings.config, production_record)
    decision["candidate_role"] = candidate.role
    decision["candidate_model"] = candidate.spec.name
    decision["production_metadata_path"] = settings.config.model_gate.production_metadata_path
    gate_path = write_model_gate_report(settings, decision)
    report_path = settings.resolve(evaluation_report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["model_gate"] = decision
    report["model_gate_report"] = str(gate_path.relative_to(settings.project_root))
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for role, result in results.items():
        log_evaluation_artifact(settings, result.extras.get("mlflow_run_id"), report_path)
    logger.info("MODEL GATE %s for candidate=%s primary_metric=%s candidate_value=%s threshold=%s",
                decision["status"], candidate.spec.name, decision["primary_metric"],
                decision["candidate_metrics"].get(decision["primary_metric"]),
                decision["absolute_quality_check"]["threshold"])
    for reason in decision["failure_reasons"]:
        logger.error("MODEL GATE rejection reason: %s", reason)
    logger.info("Production model preserved; promotion_performed=False")
    return {**decision, "report_path": str(gate_path.relative_to(settings.project_root))}


@task(name="8-register-and-promote", description="Register and promote only Model-Gate-approved candidates", retries=0)
def registry_promotion_task(results: dict[str, ModelResult], settings: Settings,
                            gate: dict[str, Any], evaluation_report_path: str) -> dict[str, Any]:
    """The only pipeline task permitted to perform registry transitions."""
    logger = get_run_logger()
    if not gate.get("passed", False):
        logger.error("Registry transition blocked: Model Gate did not pass")
        return {
            "status": "REJECTED_BY_MODEL_GATE", "registered": False,
            "promoted": False, "production_preserved": True,
            "reason": "Model Gate did not approve this candidate; registry was not called.",
        }
    candidate = results["candidate"]
    result = register_and_promote_approved_candidate(
        settings=settings, gate_decision=gate, candidate_result=candidate,
        evaluation_report_path=evaluation_report_path,
    )
    logger.info("Model Registry result: status=%s registered=%s promoted=%s model=%s version=%s run_id=%s",
                result.get("status"), result.get("registered"), result.get("promoted"),
                result.get("registered_model_name"), result.get("model_version"), result.get("run_id"))
    return result


@task(name="9-rollback-production", description="Rollback the production registry alias to a known-good approved version", retries=0)
def rollback_production_task(settings: Settings, target_version: str | int,
                            expected_current_version: str | int | None = None,
                            reason: str = "rollback to previous known-good production version") -> dict[str, Any]:
    """The only pipeline task permitted to perform a production rollback."""
    logger = get_run_logger()
    result = rollback_production_to_version(
        settings=settings, target_version=target_version,
        expected_current_version=expected_current_version, reason=reason,
    )
    logger.warning(
        "PRODUCTION ROLLBACK: before=%s promoted=%s action=%s final=%s",
        result["before_production_version"], result["promoted_version"],
        result["rollback_action"], result["final_production_version"],
    )
    return result


@flow(name="house-price-production-rollback", log_prints=True,
description="Controlled MLflow registry rollback to a known-good approved model version")
def rollback_production_flow(config_path: str | None, target_version: str,
                             expected_current_version: str | None = None,
                             reason: str = "rollback to previous known-good production version") -> dict[str, Any]:
    """Execute a registry-only production rollback through a Prefect task."""
    settings = load_config(config_path, check_paths=True)
    return rollback_production_task(
        settings=settings, target_version=target_version,
        expected_current_version=expected_current_version, reason=reason,
    )


@flow(name="house-price-ml-pipeline", log_prints=True, description="Config-driven house-price pipeline with a validation data gate")
def house_price_flow(config_path: str | None = None, data_path: str | None = None) -> dict[str, Any]:
    """Run ingestion -> validation -> gate -> feature engineering -> training -> evaluation.

    ``data_path`` is an explicit input override intended for controlled testing and
    reproducible batch runs; it never edits the configured source file.
    """
    logger = get_run_logger()
    settings = load_config(config_path, check_paths=True)
    logger.info("Starting flow with config=%s; project_root=%s; prefect_available=%s", settings.config_path, settings.project_root, PREFECT_AVAILABLE)
    frame = ingestion_task(settings, data_path)
    dataset_path = Path(data_path).expanduser().resolve() if data_path else settings.train_path
    report = validation_task(frame, settings)
    gated_frame = data_gate_task(frame, report)
    # Snapshot production metrics after the Data Gate but before this run writes
    # evaluation reports. Missing metadata with an existing model fails closed.
    production_record = load_production_record(settings)
    split = feature_engineering_task(gated_frame, settings)
    results = training_task(split, settings, report, dataset_path)
    summary = evaluation_task(gated_frame, split, results, settings, dataset_path)
    gate = model_quality_gate_task(results, settings, production_record, summary["report_path"])
    summary["model_gate"] = gate
    summary["model_gate_report"] = gate["report_path"]
    if gate["passed"]:
        registry_result = registry_promotion_task(results, settings, gate, summary["report_path"])
        summary["model_registry"] = registry_result
        summary["production_model_preserved"] = not bool(registry_result.get("promoted"))
        if registry_result.get("promoted"):
            summary["status"] = "completed"
        elif registry_result.get("status") == "DISABLED":
            summary["status"] = "completed"
        else:
            summary["status"] = "registry_promotion_failed"
    else:
        # Do not call the registry task on a rejected candidate. This ensures
        # there is no model-version registration or alias transition at all.
        summary["model_registry"] = {
            "status": "REJECTED_BY_MODEL_GATE", "registered": False,
            "promoted": False, "production_preserved": True,
            "reason": "Model Gate failed; registry task was not invoked.",
        }
        summary["production_model_preserved"] = True
        summary["status"] = "model_gate_failed"
    logger.info("Flow ended with status=%s; report=%s; model_gate=%s; registry=%s",
                summary["status"], summary["report_path"], gate["status"],
                summary["model_registry"].get("status"))
    return summary
