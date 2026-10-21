from __future__ import annotations

from fastapi.testclient import TestClient

from mlops_pipeline.config.loader import load_config
from mlops_pipeline.serving.api import create_app


class FakeApprovedModel:
    """Test double for a model already resolved from the production alias."""
    def predict(self, frame):
        # Deterministic test output; does not load any project model artifact.
        return [100.0 + float(value) for value in frame["Area_sqft"].tolist()]


METADATA = {
    "registered_model_name": "house-price-regressor",
    "model_version": "test-version-7",
    "production_alias": "production",
    "run_id": "test-approved-run",
    "model_gate_status": "PASS",
    "model_uri": "models:/house-price-regressor@production",
}


def _app(loader=None):
    return create_app(
        model_loader=loader or (lambda settings: (FakeApprovedModel(), METADATA)),
        settings_loader=load_config,
    )


def _valid_instance():
    return {
        "Area_sqft": 1000,
        "Bedrooms": 2,
        "Bathrooms": 1,
        "Age_Years": 10,
        "Distance_to_Center_km": 5.5,
        "Neighborhood_Score": 7.1,
        "Has_Parking": 1,
        "Property_Condition": "Good",
        "City_Zone": "South",
    }


def test_health_identifies_approved_model_and_version():
    with TestClient(_app()) as client:
        response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model"]["registered_model_name"] == "house-price-regressor"
    assert body["model"]["model_version"] == "test-version-7"
    assert body["model"]["production_alias"] == "production"


def test_valid_prediction_uses_model_and_returns_metadata():
    with TestClient(_app()) as client:
        response = client.post("/predict", json={"instances": [_valid_instance()]})
        metrics = client.get("/metrics")
    assert response.status_code == 200
    body = response.json()
    assert body["predictions"] == [1100.0]
    assert body["prediction_unit"] == "Price_Lakhs"
    assert body["count"] == 1
    assert body["model"]["model_version"] == "test-version-7"
    assert metrics.status_code == 200
    assert "mlops_prediction_requests_total 1.0" in metrics.text
    assert "mlops_prediction_value_bucket" in metrics.text


def test_invalid_feature_schema_returns_422_safely():
    invalid = _valid_instance()
    invalid.pop("Area_sqft")
    with TestClient(_app()) as client:
        response = client.post("/predict", json={"instances": [invalid]})
    assert response.status_code == 422
    assert "missing feature" in response.json()["detail"]


def test_invalid_top_level_request_returns_422():
    with TestClient(_app()) as client:
        response = client.post("/predict", json={"not_instances": []})
    assert response.status_code == 422


def test_model_loading_failure_health_and_prediction_fail_closed():
    def failed_loader(settings):
        raise RuntimeError("No production alias is configured in this test")

    with TestClient(_app(loader=failed_loader)) as client:
        health = client.get("/health")
        prediction = client.post("/predict", json={"instances": [_valid_instance()]})
        metrics = client.get("/metrics")
    assert health.status_code == 503
    assert health.json()["model_loaded"] is False
    assert prediction.status_code == 503
    assert "unavailable" in prediction.json()["detail"]
    assert "mlops_prediction_requests_total" in metrics.text


def test_registry_loader_resolves_production_alias_then_loads_exact_version(monkeypatch):
    import sys
    import types
    from types import SimpleNamespace
    from mlops_pipeline.serving.registry_loader import load_approved_production_model

    loaded_uris = []
    client = SimpleNamespace(
        get_model_version_by_alias=lambda name, alias: SimpleNamespace(
            version="12", run_id="approved-run-12",
            tags={"model_gate.status": "PASS", "dataset.sha256": "dataset-hash"},
        )
    )
    tracking = types.ModuleType("mlflow.tracking")
    tracking.MlflowClient = lambda: client
    fake_mlflow = types.ModuleType("mlflow")
    fake_mlflow.set_tracking_uri = lambda uri: None
    fake_mlflow.pyfunc = SimpleNamespace(load_model=lambda uri: loaded_uris.append(uri) or FakeApprovedModel())
    fake_mlflow.tracking = tracking
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
    monkeypatch.setitem(sys.modules, "mlflow.tracking", tracking)

    model, metadata = load_approved_production_model(load_config())
    assert isinstance(model, FakeApprovedModel)
    assert loaded_uris == ["models:/house-price-regressor/12"]
    assert metadata["model_version"] == "12"
    assert metadata["production_alias"] == "production"
    assert metadata["model_gate_status"] == "PASS"


def test_registry_loader_refuses_unapproved_alias_version(monkeypatch):
    import sys
    import types
    from types import SimpleNamespace
    import pytest
    from mlops_pipeline.serving.registry_loader import load_approved_production_model

    client = SimpleNamespace(
        get_model_version_by_alias=lambda name, alias: SimpleNamespace(
            version="13", run_id="unapproved-run-13", tags={"model_gate.status": "FAIL"},
        )
    )
    tracking = types.ModuleType("mlflow.tracking")
    tracking.MlflowClient = lambda: client
    fake_mlflow = types.ModuleType("mlflow")
    fake_mlflow.set_tracking_uri = lambda uri: None
    fake_mlflow.pyfunc = SimpleNamespace(load_model=lambda uri: pytest.fail("unapproved model loaded"))
    fake_mlflow.tracking = tracking
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)
    monkeypatch.setitem(sys.modules, "mlflow.tracking", tracking)

    with pytest.raises(RuntimeError, match="Model-Gate-approved"):
        load_approved_production_model(load_config())
