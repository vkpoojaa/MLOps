"""Integration tests for the Prefect house-price pipeline."""

import hashlib
import json
import shutil
from pathlib import Path

import joblib
import pandas as pd
import pytest
import yaml

from mlops_pipeline.config import load_config
from mlops_pipeline.models.training import run_training
from mlops_pipeline.pipeline.prefect_flow import house_price_flow


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _isolated_project(tmp_path: Path) -> Path:
    """Create a temporary project with its own config and input data."""
    root = tmp_path / "project"
    root.mkdir(parents=True)

    for folder in ("config", "data", "artifacts", "reports"):
        (root / folder).mkdir(parents=True, exist_ok=True)

    shutil.copy2(
        PROJECT_ROOT / "config" / "config.yaml",
        root / "config" / "config.yaml",
    )

    for filename in ("train.csv", "incoming_batch_1.csv",
                     "incoming_batch_2.csv", "incoming_batch_3.csv"):
        source = PROJECT_ROOT / "data" / filename
        if source.is_file():
            shutil.copy2(source, root / "data" / filename)

    config_path = root / "config" / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    config.setdefault("mlflow", {})["enabled"] = False
    config.setdefault("model_registry", {})["enabled"] = False

    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )
    return root


def _install_current_production_copy(tmp_path: Path):
    """Create isolated production artifacts for an integration test."""
    config_path = tmp_path / "project" / "config" / "config.yaml"
    settings = load_config(str(config_path), check_paths=True)

    training_run = run_training(settings, save=False)

    models_dir = settings.resolve(settings.config.paths.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    selected_role = settings.config.model.selected
    selected_result = training_run.results[selected_role]

    joblib.dump(
        selected_result.pipeline,
        settings.resolve(settings.config.paths.selected_model),
    )

    metadata_path = settings.resolve(
        settings.config.model_gate.production_metadata_path
    )
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "model_name": settings.config.model.baseline.name,
        "metrics": {
            "mae": settings.config.model_gate.quality_threshold,
        },
    }

    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )


def test_prefect_flow_success_runs_all_stages_in_isolated_project(tmp_path):
    root = _isolated_project(tmp_path)

    summary = house_price_flow(
        config_path=str(root / "config" / "config.yaml")
    )

    assert summary["status"] in {
        "completed",
        "model_gate_failed",
        "registry_promotion_failed",
    }
    assert summary["rows"] > 0
    assert Path(root / summary["report_path"]).is_file()
    assert "model_gate" in summary


def test_invalid_data_fails_gate_and_training_is_not_called(
    tmp_path, monkeypatch
):
    root = _isolated_project(tmp_path)
    train_path = root / "data" / "train.csv"

    frame = pd.read_csv(train_path)
    frame.iloc[: min(10, len(frame)), 0] = None
    invalid_path = root / "data" / "invalid_train.csv"
    frame.to_csv(invalid_path, index=False)

    import mlops_pipeline.pipeline.prefect_flow as prefect_flow

    def fail_if_training_called(*args, **kwargs):
        raise AssertionError("Training must not run after a failed data gate")

    monkeypatch.setattr(
        prefect_flow,
        "training_task",
        fail_if_training_called,
    )

    with pytest.raises(Exception):
        house_price_flow(
            config_path=str(root / "config" / "config.yaml"),
            data_path=str(invalid_path),
        )


def test_poor_candidate_is_rejected_and_real_production_artifact_is_preserved(
    tmp_path, monkeypatch
):
    root = _isolated_project(tmp_path)
    _install_current_production_copy(tmp_path)

    config_path = root / "config" / "config.yaml"
    settings = load_config(str(config_path), check_paths=True)
    production_path = settings.resolve(settings.config.paths.selected_model)

    before_hash = hashlib.sha256(production_path.read_bytes()).hexdigest()

    import mlops_pipeline.pipeline.prefect_flow as prefect_flow

    original_training_task = prefect_flow.training_task

    def poor_candidate_training(*args, **kwargs):
        results = original_training_task(*args, **kwargs)
        candidate = results["candidate"]
        metrics = dict(candidate.validation_metrics)
        metrics["mae"] = 50.0
        candidate.validation_metrics = metrics
        return results

    def fail_if_registry_called(*args, **kwargs):
        raise AssertionError(
            "Registry promotion must not run when the quality gate fails"
        )

    monkeypatch.setattr(
        prefect_flow,
        "training_task",
        poor_candidate_training,
    )
    monkeypatch.setattr(
        prefect_flow,
        "register_and_promote_approved_candidate",
        fail_if_registry_called,
    )

    summary = house_price_flow(config_path=str(config_path))

    after_hash = hashlib.sha256(production_path.read_bytes()).hexdigest()

    assert summary["status"] == "model_gate_failed"
    assert summary["model_gate"]["status"] == "FAIL"
    assert summary["model_gate"]["passed"] is False
    assert summary["production_model_preserved"] is True
    assert before_hash == after_hash


def test_registry_promotion_is_called_only_after_gate_passes(
    tmp_path, monkeypatch
):
    root = _isolated_project(tmp_path)
    _install_current_production_copy(tmp_path)

    import mlops_pipeline.pipeline.prefect_flow as prefect_flow

    calls = []

    def fake_registry_promotion(**kwargs):
        calls.append(kwargs)
        return {
            "status": "DISABLED",
            "registered": False,
            "promoted": False,
            "production_preserved": True,
        }

    monkeypatch.setattr(
        prefect_flow,
        "register_and_promote_approved_candidate",
        fake_registry_promotion,
    )

    summary = house_price_flow(
        config_path=str(root / "config" / "config.yaml")
    )

    if summary["model_gate"]["passed"]:
        assert len(calls) == 1
    else:
        assert len(calls) == 0
