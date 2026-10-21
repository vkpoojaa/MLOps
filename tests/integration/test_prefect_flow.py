"""Integration coverage for Phase 4 orchestration and the validation gate."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import joblib
import pandas as pd
import pytest
import yaml

from mlops_pipeline.data import DataValidationError
from mlops_pipeline.pipeline import prefect_flow

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _isolated_project(tmp_path: Path) -> Path:
    """Copy config and source CSVs so tests never alter supplied datasets/artifacts."""
    (tmp_path / "config").mkdir(parents=True)
    (tmp_path / "data" / "raw").mkdir(parents=True)

    config_path = tmp_path / "config" / "config.yaml"
    shutil.copy2(PROJECT_ROOT / "config" / "config.yaml", config_path)

    raw_cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    raw_cfg["mlflow"]["enabled"] = False
    raw_cfg["model_registry"]["enabled"] = False
    config_path.write_text(
        yaml.safe_dump(raw_cfg, sort_keys=False),
        encoding="utf-8",
    )

    for name in (
        "capstone_house_prices_train.csv",
        "capstone_incoming_batches.csv",
    ):
        shutil.copy2(
            PROJECT_ROOT / "data" / "raw" / name,
            tmp_path / "data" / "raw" / name,
        )

    return config_path


def _install_current_production_copy(tmp_path: Path) -> None:
    """Create isolated production artifacts with valid integrity metadata."""
    from mlops_pipeline.config.loader import load_config
    from mlops_pipeline.models.training import run_training

    config_path = tmp_path / "config" / "config.yaml"
    settings = load_config(config_path, check_paths=True)
    training_run = run_training(settings, save=False)

    models_dir = tmp_path / "artifacts" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    selected_role = settings.config.model.selected
    selected_result = training_run.results[selected_role]
    production_path = models_dir / "selected_model.joblib"
    joblib.dump(selected_result.pipeline, production_path)

    primary_metric = settings.config.model.evaluation.primary_metric
    metadata = {
        "model_name": settings.config.model.baseline.name,
        "metrics": {
            primary_metric: settings.config.model_gate.quality_threshold,
        },
        "artifact_sha256": _sha256(production_path),
    }

    metadata_path = (
        models_dir / "production_model_metadata.json"
    )
    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )


def test_prefect_flow_success_runs_all_stages_in_isolated_project(tmp_path):
    config_path = _isolated_project(tmp_path)
    summary = prefect_flow.house_price_flow(config_path=str(config_path))

    assert summary["status"] == "completed"
    assert summary["rows"] == 4000
    assert summary["leakage_checks_total"] > 0
    assert Path(tmp_path / summary["report_path"]).is_file()
    assert "baseline" in summary["saved_outputs"]
    assert "candidate" in summary["saved_outputs"]
    assert summary["model_gate"]["status"] == "PASS"
    assert summary["model_gate"]["production_comparison"]["status"] == "SKIPPED"
    assert summary["production_model_preserved"] is True
    assert Path(tmp_path / summary["model_gate_report"]).is_file()


def test_invalid_data_fails_gate_and_training_is_not_called(tmp_path, monkeypatch):
    config_path = _isolated_project(tmp_path)
    original = pd.read_csv(
        tmp_path / "data" / "raw" / "capstone_house_prices_train.csv"
    )
    bad_path = tmp_path / "invalid_training_data.csv"
    original.loc[0, "City_Zone"] = "NOT_A_VALID_ZONE"
    original.to_csv(bad_path, index=False)

    existing_model = tmp_path / "artifacts" / "models" / "keep-existing.txt"
    existing_model.parent.mkdir(parents=True)
    existing_model.write_text(
        "existing-model-must-survive", encoding="utf-8"
    )
    existing_report = tmp_path / "reports" / "model_evaluation.json"
    existing_report.parent.mkdir(parents=True)
    existing_report.write_text(
        "existing-report-must-survive", encoding="utf-8"
    )

    def training_must_not_run(*_args, **_kwargs):
        pytest.fail("training task was invoked after the data gate failed")

    monkeypatch.setattr(
        prefect_flow, "training_task", training_must_not_run
    )

    with pytest.raises(DataValidationError) as exc_info:
        prefect_flow.house_price_flow(
            config_path=str(config_path), data_path=str(bad_path)
        )

    assert not exc_info.value.report.passed
    assert any(result.failed for result in exc_info.value.report.results)
    assert existing_model.read_text(encoding="utf-8") == "existing-model-must-survive"
    assert existing_report.read_text(encoding="utf-8") == "existing-report-must-survive"


def test_poor_candidate_is_rejected_and_real_production_artifact_is_preserved(
    tmp_path, monkeypatch
):
    config_path = _isolated_project(tmp_path)
    _install_current_production_copy(tmp_path)

    production_path = tmp_path / "artifacts" / "models" / "selected_model.joblib"
    before = _sha256(production_path)

    def registry_must_not_run(*_args, **_kwargs):
        pytest.fail("registry/promotion was invoked after the Model Gate failed")

    monkeypatch.setattr(
        prefect_flow, "registry_promotion_task", registry_must_not_run
    )

    original_training = prefect_flow.training_task

    def controlled_poor_candidate(*args, **kwargs):
        results = original_training(*args, **kwargs)
        results["candidate"].validation_metrics["mae"] = 50.0
        return results

    monkeypatch.setattr(
        prefect_flow, "training_task", controlled_poor_candidate
    )

    summary = prefect_flow.house_price_flow(config_path=str(config_path))
    after = _sha256(production_path)

    assert summary["status"] == "model_gate_failed"
    assert summary["model_gate"]["status"] == "FAIL"
    assert summary["model_gate"]["candidate_metrics"]["mae"] == 50.0
    assert summary["model_gate"]["production_comparison"]["production_metrics"]["mae"] == pytest.approx(
        5.793117272533898
    )
    assert summary["model_gate"]["failure_reasons"]
    assert before == after


def test_registry_promotion_is_called_only_after_gate_passes(tmp_path, monkeypatch):
    config_path = _isolated_project(tmp_path)
    calls = []

    def approved_registry(results, settings, gate, report_path):
        assert gate["passed"] is True
        assert results["candidate"].extras.get("mlflow_run_id") is None
        calls.append(gate["status"])
        return {
            "status": "PROMOTED",
            "registered": True,
            "promoted": True,
            "production_preserved": False,
            "model_version": "stub-version",
        }

    monkeypatch.setattr(
        prefect_flow, "registry_promotion_task", approved_registry
    )
    summary = prefect_flow.house_price_flow(config_path=str(config_path))

    assert summary["status"] == "completed"
    assert summary["model_gate"]["passed"] is True
    assert summary["model_registry"]["promoted"] is True
    assert calls == ["PASS"]
