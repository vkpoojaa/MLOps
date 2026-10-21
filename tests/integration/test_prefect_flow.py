"""Integration coverage for Phase 4 orchestration and the validation gate."""
from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest

from mlops_pipeline.data import DataValidationError
from mlops_pipeline.pipeline import prefect_flow


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _isolated_project(tmp_path: Path) -> Path:
    """Copy config and source CSVs so tests never alter supplied datasets/artifacts."""
    (tmp_path / "config").mkdir()
    (tmp_path / "data" / "raw").mkdir(parents=True)
    shutil.copy2(PROJECT_ROOT / "config" / "config.yaml", tmp_path / "config" / "config.yaml")
    # These orchestration tests isolate flow/gate behavior from the unavailable
    # external MLflow package; tracking itself has dedicated mocked tests.
    import yaml
    cfg_path = tmp_path / "config" / "config.yaml"
    raw_cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    raw_cfg["mlflow"]["enabled"] = False
    raw_cfg["model_registry"]["enabled"] = False
    cfg_path.write_text(yaml.safe_dump(raw_cfg, sort_keys=False), encoding="utf-8")
    for name in ("capstone_house_prices_train.csv", "capstone_incoming_batches.csv"):
        shutil.copy2(PROJECT_ROOT / "data" / "raw" / name, tmp_path / "data" / "raw" / name)
    return tmp_path / "config" / "config.yaml"


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
    original = pd.read_csv(tmp_path / "data" / "raw" / "capstone_house_prices_train.csv")
    bad_path = tmp_path / "invalid_training_data.csv"
    original.loc[0, "City_Zone"] = "NOT_A_VALID_ZONE"
    original.to_csv(bad_path, index=False)

    # Existing production-like outputs must survive a failed run unchanged.
    existing_model = tmp_path / "artifacts" / "models" / "keep-existing.txt"
    existing_model.parent.mkdir(parents=True)
    existing_model.write_text("existing-model-must-survive", encoding="utf-8")
    existing_report = tmp_path / "reports" / "model_evaluation.json"
    existing_report.parent.mkdir(parents=True)
    existing_report.write_text("existing-report-must-survive", encoding="utf-8")

    def training_must_not_run(*_args, **_kwargs):
        pytest.fail("training task was invoked after the data gate failed")

    monkeypatch.setattr(prefect_flow, "training_task", training_must_not_run)
    with pytest.raises(DataValidationError) as exc_info:
        prefect_flow.house_price_flow(config_path=str(config_path), data_path=str(bad_path))

    assert not exc_info.value.report.passed
    assert any(result.failed for result in exc_info.value.report.results)
    assert existing_model.read_text(encoding="utf-8") == "existing-model-must-survive"
    assert existing_report.read_text(encoding="utf-8") == "existing-report-must-survive"


def _install_current_production_copy(tmp_path: Path):
    """Install the repository's real selected model + its evidence-derived metadata."""
    import json
    from pathlib import Path
    models_dir = tmp_path / "artifacts" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PROJECT_ROOT / "artifacts" / "models" / "selected_model.joblib",
                 models_dir / "selected_model.joblib")
    shutil.copy2(PROJECT_ROOT / "artifacts" / "models" / "production_model_metadata.json",
                 models_dir / "production_model_metadata.json")


def test_poor_candidate_is_rejected_and_real_production_artifact_is_preserved(tmp_path, monkeypatch):
    import hashlib
    config_path = _isolated_project(tmp_path)
    _install_current_production_copy(tmp_path)
    production_path = tmp_path / "artifacts" / "models" / "selected_model.joblib"
    before = hashlib.sha256(production_path.read_bytes()).hexdigest()

    def registry_must_not_run(*_args, **_kwargs):
        pytest.fail("registry/promotion was invoked after the Model Gate failed")
    monkeypatch.setattr(prefect_flow, "registry_promotion_task", registry_must_not_run)

    original_training = prefect_flow.training_task
    def controlled_poor_candidate(*args, **kwargs):
        results = original_training(*args, **kwargs)
        # Controlled poor-model scenario: override only the candidate's measured
        # gate metric after real training; never manipulate production.
        results["candidate"].validation_metrics["mae"] = 50.0
        return results
    monkeypatch.setattr(prefect_flow, "training_task", controlled_poor_candidate)

    summary = prefect_flow.house_price_flow(config_path=str(config_path))
    after = hashlib.sha256(production_path.read_bytes()).hexdigest()
    assert summary["status"] == "model_gate_failed"
    assert summary["model_gate"]["status"] == "FAIL"
    assert summary["model_gate"]["candidate_metrics"]["mae"] == 50.0
    assert summary["model_gate"]["production_comparison"]["production_metrics"]["mae"] == pytest.approx(5.793117272533898)
    assert summary["model_gate"]["failure_reasons"]
    assert before == after




def test_registry_promotion_is_called_only_after_gate_passes(tmp_path, monkeypatch):
    config_path = _isolated_project(tmp_path)
    calls = []

    def approved_registry(results, settings, gate, report_path):
        assert gate["passed"] is True
        assert results["candidate"].extras.get("mlflow_run_id") is None  # isolated registry stub
        calls.append(gate["status"])
        return {"status": "PROMOTED", "registered": True, "promoted": True,
                "production_preserved": False, "model_version": "stub-version"}

    monkeypatch.setattr(prefect_flow, "registry_promotion_task", approved_registry)
    summary = prefect_flow.house_price_flow(config_path=str(config_path))
    assert summary["status"] == "completed"
    assert summary["model_gate"]["passed"] is True
    assert summary["model_registry"]["promoted"] is True
    assert calls == ["PASS"]
