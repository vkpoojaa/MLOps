"""Resolve and load only the Phase 7 approved production registry alias.

There is deliberately no local-joblib fallback: a missing registry, alias, or
approval tag makes serving unavailable rather than silently serving a candidate.
"""
from __future__ import annotations

from typing import Any
from pathlib import Path

from ..config.loader import Settings


def load_approved_production_model(settings: Settings) -> tuple[Any, dict[str, Any]]:
    cfg = settings.config
    registry_cfg = cfg.model_registry
    if not registry_cfg.enabled:
        raise RuntimeError("Model Registry is disabled; refusing to load a local model")
    if not cfg.mlflow.enabled:
        raise RuntimeError("MLflow is disabled; production registry model cannot be resolved")

    try:
        import mlflow
        from mlflow.tracking import MlflowClient
    except ImportError as exc:
        raise RuntimeError("MLflow is required to load the approved production model") from exc

    uri = cfg.mlflow.tracking_uri
    if uri.startswith("sqlite:///"):
        db_part = uri.removeprefix("sqlite:///")
        if db_part and not Path(db_part).is_absolute():
            uri = "sqlite:///" + str((settings.project_root / db_part).resolve())
    mlflow.set_tracking_uri(uri)
    client = MlflowClient()
    registered_name = registry_cfg.registered_model_name
    alias = registry_cfg.production_alias

    try:
        version_obj = client.get_model_version_by_alias(registered_name, alias)
    except Exception as exc:
        raise RuntimeError(
            f"Cannot resolve configured production alias {registered_name!r}@{alias!r}"
        ) from exc

    tags = dict(getattr(version_obj, "tags", {}) or {})
    if tags.get("model_gate.status") != "PASS":
        raise RuntimeError("Registry production alias does not point to a Model-Gate-approved version")

    version = str(getattr(version_obj, "version", ""))
    run_id = str(getattr(version_obj, "run_id", "") or tags.get("mlflow.run_id", ""))
    if not version or not run_id:
        raise RuntimeError("Approved production registry version is missing version/run lineage")

    # Resolve the alias once, then load that exact version so a concurrent alias
    # update cannot make the loaded artifact disagree with the reported version.
    model_uri = f"models:/{registered_name}/{version}"
    try:
        model = mlflow.pyfunc.load_model(model_uri)
    except Exception as exc:
        raise RuntimeError("Failed to load model from the approved production registry alias") from exc

    metadata = {
        "registered_model_name": registered_name,
        "model_version": version,
        "production_alias": alias,
        "run_id": run_id,
        "model_gate_status": tags.get("model_gate.status"),
        "dataset_sha256": tags.get("dataset.sha256"),
        "config_sha256": tags.get("config.sha256"),
        "code_version": tags.get("code.version"),
        "model_uri": model_uri,
        "resolved_from_alias": f"models:/{registered_name}@{alias}",
    }
    return model, metadata
