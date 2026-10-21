from fastapi.testclient import TestClient

from mlops_pipeline.config.loader import load_config
from mlops_pipeline.serving.api import create_app


class FakeModel:
    def predict(self, frame):
        return [100.0 + float(v) for v in frame["Area_sqft"].tolist()]


INSTANCE = {
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


def app():
    return create_app(
        model_loader=lambda settings: (
            FakeModel(),
            {"model_version": "monitoring-test", "model_gate_status": "PASS"},
        ),
        settings_loader=load_config,
    )


def test_prometheus_metrics_change_after_real_prediction_traffic():
    with TestClient(app()) as client:
        before = client.get("/metrics").text
        response = client.post("/predict", json={"instances": [INSTANCE, {**INSTANCE, "Area_sqft": 1200}]})
        after = client.get("/metrics").text

    assert response.status_code == 200
    assert before != after
    assert 'mlops_http_requests_total{method="POST",path="/predict",status="200"} 1.0' in after
    assert "mlops_prediction_requests_total 1.0" in after
    assert "mlops_prediction_value_bucket" in after
    assert "mlops_http_request_duration_seconds_bucket" in after


def test_prometheus_metrics_capture_prediction_validation_error():
    invalid = dict(INSTANCE)
    invalid.pop("Area_sqft")
    with TestClient(app()) as client:
        response = client.post("/predict", json={"instances": [invalid]})
        metrics = client.get("/metrics").text

    assert response.status_code == 422
    assert 'mlops_http_requests_total{method="POST",path="/predict",status="422"} 1.0' in metrics
    assert "mlops_prediction_errors_total 1.0" in metrics


def test_metrics_endpoint_is_prometheus_text_and_json_compatibility_is_separate():
    with TestClient(app()) as client:
        prometheus = client.get("/metrics")
        json_metrics = client.get("/metrics/json")

    assert prometheus.headers["content-type"].startswith("text/plain")
    assert "# HELP mlops_http_requests_total" in prometheus.text
    assert json_metrics.status_code == 200
    assert json_metrics.json()["service"] == "house-price-prediction-api"
