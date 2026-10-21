from __future__ import annotations

import dataclasses
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

from mlops_pipeline.monitoring.lineage import verify_production_lineage
from mlops_pipeline.monitoring import reproducibility


def _sandbox_settings(settings, tmp_path, *, copy_production_artifacts=True):
    root = tmp_path / "project"
    (root / "config").mkdir(parents=True)
    (root / "data/raw").mkdir(parents=True)
    (root / "artifacts/models").mkdir(parents=True)
    (root / "reports").mkdir(parents=True)

    shutil.copy2(settings.config_path, root / "config/config.yaml")
    shutil.copy2(settings.train_path, root / settings.config.paths.train_data)

    if copy_production_artifacts:
        prod = settings.resolve(settings.config.paths.selected_model)
        metadata = settings.resolve(settings.config.model_gate.production_metadata_path)

        if not prod.is_file():
            raise FileNotFoundError(f"Required production model is missing: {prod}")
        if not metadata.is_file():
            raise FileNotFoundError(f"Required production metadata is missing: {metadata}")

        shutil.copy2(prod, root / settings.config.paths.selected_model)
        shutil.copy2(metadata, root / settings.config.model_gate.production_metadata_path)

    return dataclasses.replace(
        settings,
        project_root=root,
        config_path=root / "config/config.yaml",
    )


def test_lineage_audit_flags_legacy_production_missing_mlflow_lineage(settings, tmp_path):
    sandbox = _sandbox_settings(settings, tmp_path)
    report = verify_production_lineage(sandbox)

    assert report["status"] == "FAIL"
    required = next(item for item in report["checks"] if item["name"] == "required_lineage_fields")
    assert not required["passed"]
    assert "run_id" in required["observed"]
    assert (sandbox.project_root / "reports/lineage_verification.json").is_file()


def test_reproducibility_check_compares_metrics_and_writes_report(settings, tmp_path, monkeypatch):
    sandbox = _sandbox_settings(settings, tmp_path, copy_production_artifacts=False)
    metric_set = {"mae": 5.0, "rmse": 6.0, "r2": 0.8}
    result = SimpleNamespace(
        results={
            "baseline": SimpleNamespace(validation_metrics=dict(metric_set)),
            "candidate": SimpleNamespace(validation_metrics=dict(metric_set)),
        }
    )
    calls = []

    monkeypatch.setattr(reproducibility, "code_version", lambda _root: "test-commit")
    monkeypatch.setattr(
        "mlops_pipeline.models.training.run_training",
        lambda _settings, save=False: calls.append(save) or result,
    )

    report = reproducibility.check_reproducibility(sandbox)

    assert report["status"] == "PASS"
    assert len(calls) == 2 and calls == [False, False]
    assert all(item["absolute_delta"] == 0 for item in report["comparisons"])
    assert report["metric_tolerance"] == sandbox.config.reproducibility.metric_tolerance
    assert (sandbox.project_root / "reports/reproducibility_check.json").is_file()


def test_reproducibility_check_fails_outside_configured_tolerance(settings, tmp_path, monkeypatch):
    sandbox = _sandbox_settings(settings, tmp_path, copy_production_artifacts=False)
    outputs = iter([5.0, 5.01])

    def fake_training(_settings, save=False):
        mae = next(outputs)
        return SimpleNamespace(
            results={
                role: SimpleNamespace(
                    validation_metrics={"mae": mae, "rmse": 6.0, "r2": 0.8}
                )
                for role in ("baseline", "candidate")
            }
        )

    monkeypatch.setattr(reproducibility, "code_version", lambda _root: "test-commit")
    monkeypatch.setattr("mlops_pipeline.models.training.run_training", fake_training)

    report = reproducibility.check_reproducibility(sandbox)

    assert report["status"] == "FAIL"
    assert any(not item["passed"] for item in report["comparisons"])
