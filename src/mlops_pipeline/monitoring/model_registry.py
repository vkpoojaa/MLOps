"""Gate-controlled MLflow Model Registry registration and production promotion.

Uses MLflow model-version aliases (not deprecated model stages). Registration and
promotion are callable only after the Model Quality Gate passes. The pipeline's
MLflow run is the source of the registered model version.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config.loader import Settings
from .mlflow_tracking import load_mlflow


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _update_json_report(path: Path, updates: dict[str, Any]) -> None:
    if not path.is_file():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(updates)
    _atomic_write(path, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"))



def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _version_metadata(client, settings: Settings, version_obj: Any, *, production_alias: str,
                     artifact_path: str) -> dict[str, Any]:
    """Build production metadata from the registry version and its source run.

    Rollback deliberately uses registry metadata only; it never downloads or copies
    the model artifact back into the repository.
    """
    version = str(getattr(version_obj, "version", ""))
    run_id = str(getattr(version_obj, "run_id", "") or "")
    if not version or not run_id:
        raise RuntimeError("Rollback target is missing registry version/run lineage")
    tags = dict(getattr(version_obj, "tags", {}) or {})
    if tags.get("model_gate.status") != "PASS":
        raise RuntimeError("Rollback target is not Model-Gate-approved")
    run = client.get_run(run_id)
    run_tags = dict(getattr(run.data, "tags", {}) or {})
    params = dict(getattr(run.data, "params", {}) or {})
    metrics = dict(getattr(run.data, "metrics", {}) or {})
    def lineage(key: str, default: Any = None):
        return run_tags.get(key, params.get(key, tags.get(key, default)))
    model_name = tags.get("model_name", "registry-approved-production")
    artifact_sha = tags.get("artifact.sha256")
    if not artifact_sha:
        raise RuntimeError("Rollback target is missing artifact.sha256 registry tag")
    return {
        "model_name": model_name,
        "metrics": {
            "mae": float(metrics.get("MAE", metrics.get("mae", float("nan")))),
            "rmse": float(metrics.get("RMSE", metrics.get("rmse", float("nan")))),
            "r2": float(metrics.get("R2", metrics.get("r2", float("nan")))),
        },
        "primary_metric": settings.config.model.evaluation.primary_metric,
        "artifact_path": artifact_path,
        "artifact_sha256": artifact_sha,
        "registered_model_name": settings.config.model_registry.registered_model_name,
        "model_version": version,
        "production_alias": production_alias,
        "candidate_alias": settings.config.model_registry.candidate_alias,
        "run_id": run_id,
        "mlflow_run_id": run_id,
        "dataset_identifier": lineage("dataset.identifier"),
        "dataset_sha256": lineage("dataset.sha256"),
        "config_path": lineage("config.path"),
        "config_sha256": lineage("config.sha256"),
        "code_version": lineage("code.version"),
        "model_gate_status": tags.get("model_gate.status"),
        "promoted_at_utc": tags.get("promoted_at_utc", datetime.now(timezone.utc).isoformat()),
        "production_mirror_status": "registry_authoritative_after_rollback",
        "registry_model_uri": f"models:/{settings.config.model_registry.registered_model_name}/{version}",
    }


def rollback_production_to_version(
    *, settings: Settings, target_version: str | int, expected_current_version: str | int | None = None,
    reason: str = "rollback to previous known-good production version",
) -> dict[str, Any]:
    """Rollback production by moving the MLflow production alias to a prior version.

    This is the Phase 12 rollback primitive. It changes only the registry alias;
    it never copies, downloads, or replaces a model file. The serving layer resolves
    the alias at startup and then loads the exact resolved registry version.
    """
    registry_cfg = settings.config.model_registry
    if not registry_cfg.enabled or not settings.config.mlflow.enabled:
        raise RuntimeError("Rollback requires MLflow Model Registry and tracking to be enabled")
    target = str(target_version)
    if not target.isdigit() or int(target) < 1:
        raise ValueError(f"Invalid rollback target version: {target_version!r}")
    mlflow = load_mlflow(settings)
    if mlflow is None:
        raise RuntimeError("MLflow is required for production rollback")
    client = mlflow.tracking.MlflowClient()
    name = registry_cfg.registered_model_name
    alias = registry_cfg.production_alias
    try:
        before_obj = client.get_model_version_by_alias(name, alias)
    except Exception as exc:
        raise RuntimeError(f"Cannot resolve current production alias {name!r}@{alias!r}") from exc
    before = str(before_obj.version)
    if expected_current_version is not None and before != str(expected_current_version):
        raise RuntimeError(
            f"Rollback safety check failed: expected production version {expected_current_version}, observed {before}"
        )
    if before == target:
        raise RuntimeError(f"Rollback target {target} is already the production version")
    try:
        target_obj = client.get_model_version(name, target)
    except Exception as exc:
        raise RuntimeError(f"Rollback target version {target} does not exist in registry") from exc
    tags = dict(getattr(target_obj, "tags", {}) or {})
    if tags.get("model_gate.status") != "PASS":
        raise RuntimeError(f"Rollback target version {target} is not Model-Gate-approved")

    # The alias switch is the only production transition. No local model artifact is touched.
    client.set_registered_model_alias(name, alias, target)
    try:
        final_obj = client.get_model_version_by_alias(name, alias)
        final = str(final_obj.version)
        if final != target:
            raise RuntimeError(f"Registry alias verification failed: expected {target}, observed {final}")
        metadata_path = settings.resolve(settings.config.model_gate.production_metadata_path)
        metadata = _version_metadata(
            client, settings, target_obj, production_alias=alias,
            artifact_path=str(settings.config.paths.selected_model),
        )
        _atomic_write(metadata_path, (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    except Exception:
        # Restore the alias if post-transition metadata/audit work fails.
        client.set_registered_model_alias(name, alias, before)
        raise

    record = {
        "event": "PRODUCTION_ROLLBACK",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "before_production_version": before,
        "promoted_version": before,
        "rollback_target_version": target,
        "rollback_action": "set_production_alias",
        "final_production_version": final,
        "registered_model_name": name,
        "production_alias": alias,
        "reason": reason,
        "target_run_id": str(getattr(target_obj, "run_id", "") or tags.get("mlflow.run_id", "")),
        "model_file_copied": False,
    }
    audit_path = settings.resolve(settings.config.paths.reports_dir) / "rollback_history.jsonl"
    _append_jsonl(audit_path, record)
    latest_path = audit_path.with_name("rollback_latest.json")
    _atomic_write(latest_path, (json.dumps(record, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    return {**record, "audit_path": str(audit_path.relative_to(settings.project_root)), "metadata_path": str(settings.config.model_gate.production_metadata_path)}


def register_and_promote_approved_candidate(
    *, settings: Settings, gate_decision: dict[str, Any], candidate_result,
    evaluation_report_path: str | None = None,
) -> dict[str, Any]:
    """Register and promote a candidate only after an explicit passing gate.

    Production alias is the sole production artifact authority. The local
    ``selected_model.joblib`` is not modified by promotion and is not a serving
    path; production metadata records the registry model URI and audit lineage.
    """
    if not gate_decision.get("passed", False):
        return {
            "status": "REJECTED_BY_MODEL_GATE",
            "registered": False,
            "promoted": False,
            "production_preserved": True,
            "reason": "Model Gate did not approve this candidate; registry was not called.",
        }

    registry_cfg = settings.config.model_registry
    if not registry_cfg.enabled:
        return {
            "status": "DISABLED", "registered": False, "promoted": False,
            "production_preserved": True,
            "reason": "Model Registry is disabled in config.yaml.",
        }
    if not settings.config.mlflow.enabled:
        raise RuntimeError("Model Registry requires MLflow tracking to be enabled")

    run_id = candidate_result.extras.get("mlflow_run_id")
    if not run_id:
        raise RuntimeError("Approved candidate has no pipeline-created MLflow run ID; refusing registration")
    lineage_requirements = {
        "dataset_identifier": candidate_result.extras.get("dataset_identifier"),
        "dataset_sha256": candidate_result.extras.get("dataset_sha256"),
        "config_path": candidate_result.extras.get("config_path"),
        "config_sha256": candidate_result.extras.get("config_sha256"),
        "code_version": candidate_result.extras.get("code_version"),
    }
    missing_lineage = [key for key, value in lineage_requirements.items()
                       if value in (None, "", "unavailable-no-git-commit")]
    if missing_lineage:
        raise RuntimeError(
            "Approved candidate has incomplete production lineage; refusing registration/promotion. "
            f"Missing or unavailable: {', '.join(missing_lineage)}. Run from a Git checkout or set GIT_COMMIT."
        )

    mlflow = load_mlflow(settings)
    if mlflow is None:
        raise RuntimeError("Model Registry is enabled but MLflow tracking is unavailable")

    cfg = settings.config
    registry_name = registry_cfg.registered_model_name
    candidate_alias = registry_cfg.candidate_alias
    production_alias = registry_cfg.production_alias
    artifact_root = cfg.mlflow.artifact_subdirectory.strip("/")
    model_uri = f"runs:/{run_id}/{artifact_root}/model"
    logger_info = {
        "registered_model_name": registry_name,
        "model_uri": model_uri,
        "run_id": run_id,
    }

    candidate_path = settings.resolve(cfg.paths.models_dir) / f"{candidate_result.spec.name}.joblib"
    if not candidate_path.is_file():
        raise FileNotFoundError(f"Candidate pipeline artifact missing; refusing promotion: {candidate_path}")

    # Registration creates a version linked to the existing training run; it
    # does not create another MLflow run.
    version_obj = mlflow.register_model(
        model_uri=model_uri,
        name=registry_name,
        await_registration_for=registry_cfg.await_registration_for_seconds,
    )
    version = str(version_obj.version)
    client = mlflow.tracking.MlflowClient()
    client.set_registered_model_alias(registry_name, candidate_alias, version)
    for key, value in {
        "model_gate.status": "PASS",
        "model_role": "candidate",
        "mlflow.run_id": run_id,
        "dataset.sha256": candidate_result.extras.get("dataset_sha256", "unavailable"),
        "config.sha256": candidate_result.extras.get("config_sha256", "unavailable"),
        "code.version": candidate_result.extras.get("code_version", "unavailable"),
        "model_name": candidate_result.spec.name,
        "artifact.sha256": _sha256(candidate_path),
        "promoted_at_utc": datetime.now(timezone.utc).isoformat(),
    }.items():
        client.set_model_version_tag(registry_name, version, key, str(value))

    metadata_path = settings.resolve(cfg.model_gate.production_metadata_path)
    gate_report_path = settings.resolve(cfg.paths.reports_dir) / "model_gate.json"
    report_path = settings.resolve(evaluation_report_path) if evaluation_report_path else None

    # The MLflow registry is the production artifact authority. The pipeline no
    # longer copies candidate model bytes into selected_model.joblib during
    # promotion. This removes a second production artifact transition.
    old_metadata = metadata_path.read_bytes() if metadata_path.is_file() else None
    candidate_hash = _sha256(candidate_path)
    primary_metric = cfg.model.evaluation.primary_metric
    metadata = {
        "model_name": candidate_result.spec.name,
        "metrics": {key: float(value) for key, value in candidate_result.validation_metrics.items()},
        "primary_metric": primary_metric,
        "artifact_path": f"models:/{registry_name}/{version}",
        "artifact_sha256": candidate_hash,
        "registered_model_name": registry_name,
        "model_version": version,
        "production_alias": production_alias,
        "candidate_alias": candidate_alias,
        "run_id": run_id,
        "mlflow_run_id": run_id,
        "dataset_identifier": candidate_result.extras.get("dataset_identifier"),
        "dataset_sha256": candidate_result.extras.get("dataset_sha256"),
        "config_path": str(settings.config_path.relative_to(settings.project_root)),
        "config_sha256": candidate_result.extras.get("config_sha256"),
        "code_version": candidate_result.extras.get("code_version"),
        "model_gate_status": "PASS",
        "promoted_at_utc": datetime.now(timezone.utc).isoformat(),
        "production_mirror_status": "registry_authoritative",
        "registry_model_uri": f"models:/{registry_name}/{version}",
    }

    try:
        _atomic_write(metadata_path, (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode("utf-8"))
        client.set_registered_model_alias(registry_name, production_alias, version)
    except Exception:
        # Restore only audit metadata; the production model itself was never
        # copied or changed locally. The registry alias is the sole transition.
        if old_metadata is None:
            metadata_path.unlink(missing_ok=True)
        else:
            _atomic_write(metadata_path, old_metadata)
        raise

    result = {
        "status": "PROMOTED",
        "registered": True,
        "promoted": True,
        "production_preserved": False,
        **logger_info,
        "model_version": version,
        "candidate_alias": candidate_alias,
        "production_alias": production_alias,
        "production_artifact": f"models:/{registry_name}/{version}",
        "production_artifact_sha256": candidate_hash,
        "primary_metric": primary_metric,
        "candidate_metrics": metadata["metrics"],
    }

    gate_decision["promotion_performed"] = True
    gate_decision["production_model_replaced"] = True
    gate_decision["production_model_preserved"] = False
    gate_decision["candidate_disposition"] = "promoted_to_production"
    gate_decision["registry"] = result
    _update_json_report(gate_report_path, gate_decision)
    if report_path:
        _update_json_report(report_path, {"model_gate": gate_decision, "model_registry": result})
    return result
