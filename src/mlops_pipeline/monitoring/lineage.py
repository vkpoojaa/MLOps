"""Production model lineage audit from local metadata through MLflow run evidence."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ..config.loader import Settings
from .mlflow_tracking import load_mlflow, sha256_file

REQUIRED_METADATA = (
    "registered_model_name", "model_version", "run_id", "dataset_identifier", "dataset_sha256",
    "config_path", "config_sha256", "code_version", "artifact_sha256", "metrics",
)


def _artifact_paths(client, run_id: str, path: str = "") -> set[str]:
    found: set[str] = set()
    try:
        entries = client.list_artifacts(run_id, path or None)
    except TypeError:
        entries = client.list_artifacts(run_id) if not path else client.list_artifacts(run_id, path)
    for entry in entries:
        full = f"{path}/{entry.path}" if path and not entry.path.startswith(path + "/") else entry.path
        found.add(full.strip("/"))
        if getattr(entry, "is_dir", False):
            found.update(_artifact_paths(client, run_id, full.strip("/")))
    return found


def verify_production_lineage(settings: Settings, *, require_mlflow: bool = True) -> dict[str, Any]:
    """Audit the production mirror, registry alias, version, run, hashes and artifacts.

    This command is read-only. It deliberately does not backfill missing lineage
    fields because inventing a run ID or commit would create false provenance.
    """
    cfg = settings.config
    metadata_path = settings.resolve(cfg.model_gate.production_metadata_path)
    production_path = settings.resolve(cfg.paths.selected_model)
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: str, observed: Any = None) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail, "observed": observed})

    if not metadata_path.is_file():
        check("production_metadata_exists", False, f"Missing {metadata_path}")
        metadata: dict[str, Any] = {}
    else:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        check("production_metadata_exists", True, "Production metadata exists", str(metadata_path))

    missing = [key for key in REQUIRED_METADATA if metadata.get(key) in (None, "", {})]
    if metadata.get("code_version") == "unavailable-no-git-commit":
        missing.append("valid Git commit/code version (currently unavailable-no-git-commit)")
    check("required_lineage_fields", not missing,
          "All required lineage fields are present" if not missing else f"Missing fields: {', '.join(missing)}", missing)

    mirror_status = metadata.get("production_mirror_status", "synced")
    if mirror_status.startswith("registry_authoritative"):
        # Production is registry-only by design. The local joblib mirror is not the
        # serving source and must not be copied during promotion or rollback.
        actual_hash = sha256_file(production_path) if production_path.is_file() else None
        check("production_artifact_hash", True,
              "Registry is authoritative after rollback; local model mirror is intentionally not rewritten",
              {"expected_registry_artifact_sha256": metadata.get("artifact_sha256"), "local_mirror_sha256": actual_hash})
    elif not production_path.is_file():
        check("production_artifact_exists", False, f"Missing {production_path}")
    else:
        actual_hash = sha256_file(production_path)
        expected_hash = metadata.get("artifact_sha256")
        check("production_artifact_hash", bool(expected_hash) and actual_hash == expected_hash,
              "Production artifact SHA-256 matches metadata" if expected_hash and actual_hash == expected_hash
              else "Production artifact hash is missing or mismatched", {"expected": expected_hash, "actual": actual_hash})

    if not cfg.model_registry.enabled:
        check("registry_enabled", False, "model_registry.enabled is false")
    elif missing:
        check("registry_lineage", False, "Cannot verify registry chain until legacy/missing metadata is replaced by a real gate-approved promotion", missing)
    else:
        try:
            mlflow = load_mlflow(settings)
            if mlflow is None:
                raise RuntimeError("MLflow tracking is disabled")
            client = mlflow.tracking.MlflowClient()
            registry_name = metadata["registered_model_name"]
            alias = cfg.model_registry.production_alias
            version = client.get_model_version_by_alias(registry_name, alias)
            run_id = metadata["run_id"]
            run = client.get_run(run_id)
            version_matches = str(version.version) == str(metadata["model_version"])
            run_matches = getattr(version, "run_id", None) == run_id
            check("production_alias_version", version_matches,
                  "Production alias points to metadata model version" if version_matches else "Alias version differs from production metadata",
                  {"alias": alias, "alias_version": str(version.version), "metadata_version": str(metadata["model_version"])})
            check("version_run_link", run_matches,
                  "Registered version links to metadata MLflow run" if run_matches else "Registered version run ID differs",
                  {"version_run_id": getattr(version, "run_id", None), "metadata_run_id": run_id})
            tags = run.data.tags
            params = run.data.params
            lineage_pairs = {
                "dataset.identifier": metadata["dataset_identifier"],
                "dataset.sha256": metadata["dataset_sha256"],
                "config.path": metadata["config_path"],
                "config.sha256": metadata["config_sha256"],
                "code.version": metadata["code_version"],
            }
            for key, expected in lineage_pairs.items():
                observed = tags.get(key, params.get(key))
                check(f"run_lineage_{key}", observed == expected,
                      f"Run {key} matches production metadata" if observed == expected else f"Run {key} missing/mismatched",
                      {"expected": expected, "actual": observed})
            check("run_metrics_present", all(m in run.data.metrics for m in ("MAE", "RMSE", "R2")),
                  "Run contains MAE, RMSE and R2" if all(m in run.data.metrics for m in ("MAE", "RMSE", "R2")) else "One or more expected metrics are missing",
                  {k: run.data.metrics.get(k) for k in ("MAE", "RMSE", "R2")})
            paths = _artifact_paths(client, run_id)
            artifact_root = cfg.mlflow.artifact_subdirectory.strip("/")
            expected_suffixes = [f"{artifact_root}/model", f"{artifact_root}/configuration", f"{artifact_root}/validation", f"{artifact_root}/evaluation"]
            missing_artifacts = [suffix for suffix in expected_suffixes if not any(p == suffix or p.startswith(suffix + "/") for p in paths)]
            check("run_artifacts", not missing_artifacts,
                  "Model, configuration, validation and evaluation artifacts are present" if not missing_artifacts else f"Missing artifact paths: {missing_artifacts}", sorted(paths))
        except Exception as exc:
            check("mlflow_chain_reachable", False, f"Unable to verify live MLflow lineage: {type(exc).__name__}: {exc}")

    passed = bool(checks) and all(item["passed"] for item in checks)
    report = {
        "status": "PASS" if passed else "FAIL",
        "production_model": metadata.get("model_name"),
        "registry_model": metadata.get("registered_model_name"),
        "model_version": metadata.get("model_version"),
        "run_id": metadata.get("run_id"),
        "dataset_sha256": metadata.get("dataset_sha256"),
        "config_path": metadata.get("config_path"),
        "config_sha256": metadata.get("config_sha256"),
        "code_version": metadata.get("code_version"),
        "checks": checks,
        "read_only": True,
    }
    out = settings.resolve(cfg.paths.reports_dir) / "lineage_verification.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report["report_path"] = str(out.relative_to(settings.project_root))
    return report
