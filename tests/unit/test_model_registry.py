from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from mlops_pipeline.config import load_config
from mlops_pipeline.monitoring import model_registry

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class FakeClient:
    def __init__(self, fail_production_alias=False):
        self.aliases = []
        self.tags = []
        self.fail_production_alias = fail_production_alias

    def set_registered_model_alias(self, name, alias, version):
        if alias == "production" and self.fail_production_alias:
            raise RuntimeError("controlled alias transition failure")
        self.aliases.append((name, alias, str(version)))

    def set_model_version_tag(self, name, version, key, value):
        self.tags.append((name, str(version), key, value))


def _settings(tmp_path: Path):
    (tmp_path / "config").mkdir(parents=True)
    (tmp_path / "data" / "raw").mkdir(parents=True)
    for filename in ("capstone_house_prices_train.csv", "capstone_incoming_batches.csv"):
        shutil.copy2(PROJECT_ROOT / "data" / "raw" / filename, tmp_path / "data" / "raw" / filename)
    raw = yaml.safe_load((PROJECT_ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))
    raw["mlflow"]["enabled"] = True
    raw["model_registry"]["enabled"] = True
    config_path = tmp_path / "config" / "config.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    settings = load_config(config_path, check_paths=True)
    models_dir = settings.resolve(settings.config.paths.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / "gradient_boosting_candidate.joblib").write_bytes(b"candidate-pipeline-bytes")
    production_path = settings.resolve(settings.config.paths.selected_model)
    production_path.parent.mkdir(parents=True, exist_ok=True)
    production_path.write_bytes(b"old-production-pipeline")
    metadata_path = settings.resolve(settings.config.model_gate.production_metadata_path)
    metadata_path.write_text(json.dumps({"model_name": "old-production", "metrics": {"mae": 5.7}, "artifact_sha256": "old-hash"}), encoding="utf-8")
    reports = settings.resolve(settings.config.paths.reports_dir)
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "model_gate.json").write_text(json.dumps({"status": "PASS", "passed": True}), encoding="utf-8")
    (reports / "model_evaluation.json").write_text(json.dumps({"status": "evaluated"}), encoding="utf-8")
    candidate = SimpleNamespace(
        spec=SimpleNamespace(name="gradient_boosting_candidate", algorithm="gradient_boosting"),
        validation_metrics={"mae": 5.5, "rmse": 7.1, "r2": 0.9},
        extras={"mlflow_run_id": "pipeline-run-123", "dataset_identifier": "train.csv", "dataset_sha256": "data-hash", "config_path": "config/config.yaml", "config_sha256": "config-hash", "code_version": "commit-abc"},
    )
    gate = {"passed": True, "status": "PASS", "candidate_metrics": candidate.validation_metrics.copy(), "promotion_performed": False}
    return settings, candidate, gate


def _fake_mlflow(monkeypatch, client):
    calls = {"register_model": []}

    def register_model(**kwargs):
        calls["register_model"].append(kwargs)
        return SimpleNamespace(version="7")

    fake = SimpleNamespace(
        register_model=register_model,
        tracking=SimpleNamespace(MlflowClient=lambda: client),
        calls=calls,
    )
    monkeypatch.setattr(model_registry, "load_mlflow", lambda settings: fake)
    return fake


def test_failed_gate_never_loads_mlflow_or_registers(tmp_path, monkeypatch):
    settings, candidate, gate = _settings(tmp_path)
    gate["passed"] = False
    monkeypatch.setattr(model_registry, "load_mlflow", lambda _settings: pytest.fail("MLflow must not be touched"))
    before = settings.resolve(settings.config.paths.selected_model).read_bytes()
    result = model_registry.register_and_promote_approved_candidate(
        settings=settings, gate_decision=gate, candidate_result=candidate,
        evaluation_report_path="reports/model_evaluation.json",
    )
    assert result["status"] == "REJECTED_BY_MODEL_GATE"
    assert result["registered"] is False and result["promoted"] is False
    assert settings.resolve(settings.config.paths.selected_model).read_bytes() == before


def test_approved_candidate_is_registered_linked_and_promoted(tmp_path, monkeypatch):
    settings, candidate, gate = _settings(tmp_path)
    client = FakeClient()
    fake = _fake_mlflow(monkeypatch, client)
    result = model_registry.register_and_promote_approved_candidate(
        settings=settings, gate_decision=gate, candidate_result=candidate,
        evaluation_report_path="reports/model_evaluation.json",
    )
    assert result["status"] == "PROMOTED"
    assert fake.calls["register_model"] == [{
        "model_uri": "runs:/pipeline-run-123/phase5/model",
        "name": "house-price-regressor",
        "await_registration_for": 120,
    }]
    assert result["run_id"] == "pipeline-run-123"
    assert result["model_version"] == "7"
    assert (settings.config.model_registry.registered_model_name, "candidate", "7") in client.aliases
    assert (settings.config.model_registry.registered_model_name, "production", "7") in client.aliases
    assert any(tag[2:] == ("mlflow.run_id", "pipeline-run-123") for tag in client.tags)
    production_path = settings.resolve(settings.config.paths.selected_model)
    assert production_path.read_bytes() == b"old-production-pipeline"
    metadata = json.loads(settings.resolve(settings.config.model_gate.production_metadata_path).read_text())
    assert metadata["model_version"] == "7" and metadata["run_id"] == "pipeline-run-123"
    assert metadata["production_mirror_status"] == "registry_authoritative"
    assert metadata["artifact_path"] == "models:/house-price-regressor/7"
    assert gate["promotion_performed"] is True
    report = json.loads(settings.resolve("reports/model_evaluation.json").read_text())
    assert report["model_registry"]["status"] == "PROMOTED"


def test_registry_alias_failure_restores_existing_production_metadata(tmp_path, monkeypatch):
    settings, candidate, gate = _settings(tmp_path)
    client = FakeClient(fail_production_alias=True)
    _fake_mlflow(monkeypatch, client)
    production_path = settings.resolve(settings.config.paths.selected_model)
    metadata_path = settings.resolve(settings.config.model_gate.production_metadata_path)
    old_model, old_metadata = production_path.read_bytes(), metadata_path.read_bytes()
    with pytest.raises(RuntimeError, match="controlled alias transition failure"):
        model_registry.register_and_promote_approved_candidate(
            settings=settings, gate_decision=gate, candidate_result=candidate,
            evaluation_report_path="reports/model_evaluation.json",
        )
    assert production_path.read_bytes() == old_model
    assert metadata_path.read_bytes() == old_metadata
    assert not any(alias == "production" for _, alias, _ in client.aliases)


def test_approved_candidate_without_git_lineage_is_not_registered(settings, monkeypatch, tmp_path):
    from mlops_pipeline.monitoring import model_registry
    from types import SimpleNamespace
    candidate = SimpleNamespace(
        spec=SimpleNamespace(name="candidate"),
        validation_metrics={"mae": 5.0},
        extras={"mlflow_run_id": "pipeline-run", "dataset_identifier": "train.csv",
                "dataset_sha256": "data-hash", "config_path": "config/config.yaml",
                "config_sha256": "config-hash", "code_version": "unavailable-no-git-commit"},
    )
    gate = {"passed": True}
    monkeypatch.setattr(model_registry, "load_mlflow", lambda _settings: pytest.fail("must reject before registry access"))
    with pytest.raises(RuntimeError, match="incomplete production lineage"):
        model_registry.register_and_promote_approved_candidate(
            settings=settings, gate_decision=gate, candidate_result=candidate)
