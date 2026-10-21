from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

after_import = None
from mlops_pipeline.monitoring import model_registry
from mlops_pipeline.serving.registry_loader import load_approved_production_model

from test_model_registry import _settings


class Version:
    def __init__(self, version, run_id, tags):
        self.version = str(version)
        self.run_id = run_id
        self.tags = dict(tags)


class Run:
    def __init__(self, tags, metrics):
        self.data = SimpleNamespace(tags=dict(tags), params={}, metrics=dict(metrics))


class RollbackClient:
    def __init__(self):
        self.versions = {
            "1": Version("1", "run-v1", {
                "model_gate.status": "PASS", "mlflow.run_id": "run-v1",
                "model_name": "house-price-v1", "artifact.sha256": "hash-v1",
            }),
        }
        self.runs = {
            "run-v1": Run({
                "dataset.identifier": "train-v1.csv", "dataset.sha256": "data-v1",
                "config.path": "config/config.yaml", "config.sha256": "config-v1",
                "code.version": "commit-v1",
            }, {"MAE": 5.1, "RMSE": 6.8, "R2": 0.91}),
        }
        self.aliases = {"production": "1"}
        self.alias_history = []

    def get_model_version_by_alias(self, name, alias):
        return self.versions[self.aliases[alias]]

    def get_model_version(self, name, version):
        return self.versions[str(version)]

    def get_run(self, run_id):
        return self.runs[run_id]

    def set_registered_model_alias(self, name, alias, version):
        self.aliases[alias] = str(version)
        self.alias_history.append((alias, str(version)))

    def set_model_version_tag(self, name, version, key, value):
        self.versions[str(version)].tags[key] = str(value)


def test_v1_v2_rollback_restores_v1_via_registry_without_copying_model(tmp_path, monkeypatch):
    settings, candidate, gate = _settings(tmp_path)
    client = RollbackClient()

    # Existing Phase 7 promotion machinery creates v2 and moves production to v2.
    def register_model(**kwargs):
        client.versions["2"] = Version("2", "run-v2", {
            "model_gate.status": "PASS", "mlflow.run_id": "run-v2",
        })
        client.runs["run-v2"] = Run({
            "dataset.identifier": "train-v2.csv", "dataset.sha256": "data-v2",
            "config.path": "config/config.yaml", "config.sha256": "config-v2",
            "code.version": "commit-v2",
        }, {"MAE": 4.9, "RMSE": 6.5, "R2": 0.93})
        return SimpleNamespace(version="2")

    fake_mlflow = SimpleNamespace(
        register_model=register_model,
        tracking=SimpleNamespace(MlflowClient=lambda: client),
    )
    monkeypatch.setattr(model_registry, "load_mlflow", lambda settings: fake_mlflow)

    promotion = model_registry.register_and_promote_approved_candidate(
        settings=settings, gate_decision=gate, candidate_result=candidate,
    )
    assert promotion["model_version"] == "2"
    assert client.aliases["production"] == "2"

    production_path = settings.resolve(settings.config.paths.selected_model)
    v2_bytes = production_path.read_bytes()

    rollback = model_registry.rollback_production_to_version(
        settings=settings,
        target_version="1",
        expected_current_version="2",
        reason="restore previous known-good v1 after v2 deployment test",
    )

    assert rollback["before_production_version"] == "2"
    assert rollback["promoted_version"] == "2"
    assert rollback["rollback_action"] == "set_production_alias"
    assert rollback["final_production_version"] == "1"
    assert rollback["model_file_copied"] is False
    assert client.aliases["production"] == "1"
    # The rollback did not copy v1 over the local v2 mirror.
    assert production_path.read_bytes() == v2_bytes

    latest = json.loads((settings.resolve(settings.config.paths.reports_dir) / "rollback_latest.json").read_text())
    assert latest["before_production_version"] == "2"
    assert latest["promoted_version"] == "2"
    assert latest["final_production_version"] == "1"
    history = (settings.resolve(settings.config.paths.reports_dir) / "rollback_history.jsonl").read_text().strip().splitlines()
    assert len(history) == 1


def test_rollback_requires_expected_current_version(tmp_path, monkeypatch):
    settings, _, _ = _settings(tmp_path)
    client = RollbackClient()
    fake_mlflow = SimpleNamespace(tracking=SimpleNamespace(MlflowClient=lambda: client))
    monkeypatch.setattr(model_registry, "load_mlflow", lambda settings: fake_mlflow)
    with pytest.raises(RuntimeError, match="expected production version 2, observed 1"):
        model_registry.rollback_production_to_version(
            settings=settings, target_version="1", expected_current_version="2"
        )
    assert client.aliases["production"] == "1"


def test_serving_loader_uses_rolled_back_production_alias_version(monkeypatch, tmp_path):
    settings, _, _ = _settings(tmp_path)
    client = RollbackClient()
    client.aliases["production"] = "1"
    loaded_uris = []

    class Model:
        def predict(self, frame):
            return [42.0] * len(frame)

    fake_mlflow = SimpleNamespace(
        set_tracking_uri=lambda uri: None,
        tracking=SimpleNamespace(MlflowClient=lambda: client),
        pyfunc=SimpleNamespace(load_model=lambda uri: loaded_uris.append(uri) or Model()),
    )
    tracking = SimpleNamespace(MlflowClient=lambda: client)
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
    monkeypatch.setitem(sys.modules, "mlflow.tracking", tracking)

    model, metadata = load_approved_production_model(settings)
    assert model is not None
    assert metadata["model_version"] == "1"
    assert metadata["production_alias"] == "production"
    assert loaded_uris == ["models:/house-price-regressor/1"]


def test_api_health_reports_rolled_back_model_version(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    from mlops_pipeline.serving.api import create_app

    settings, _, _ = _settings(tmp_path)
    client = RollbackClient()
    client.aliases["production"] = "1"

    class Model:
        def predict(self, frame):
            return [42.0] * len(frame)

    loaded = {"model": Model(), "metadata": {
        "registered_model_name": "house-price-regressor",
        "model_version": "1", "production_alias": "production", "model_gate_status": "PASS",
    }}
    app = create_app(model_loader=lambda _settings: (loaded["model"], loaded["metadata"]), settings_loader=lambda: settings)
    with TestClient(app) as api:
        response = api.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["model"]["model_version"] == "1"
    assert body["model"]["production_alias"] == "production"


def test_rollback_is_exposed_as_prefect_pipeline_task():
    from mlops_pipeline.pipeline.prefect_flow import rollback_production_flow, rollback_production_task
    assert rollback_production_flow.__name__ == "rollback_production_flow"
    assert rollback_production_task.__name__ == "rollback_production_task"
    source = Path("src/mlops_pipeline/pipeline/prefect_flow.py").read_text(encoding="utf-8")
    assert 'name="9-rollback-production"' in source
