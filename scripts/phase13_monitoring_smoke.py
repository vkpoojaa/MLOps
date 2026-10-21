"""Generate real ASGI API traffic and verify Prometheus scrape changes."""
from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from mlops_pipeline.config.loader import load_config
from mlops_pipeline.serving.api import create_app


class DemoModel:
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


def metric_value(text: str, metric: str) -> float:
    prefix = metric + " "
    for line in text.splitlines():
        if line.startswith(prefix):
            return float(line.split()[-1])
    raise AssertionError(f"metric {metric!r} not found")


def main() -> None:
    app = create_app(
        model_loader=lambda settings: (
            DemoModel(),
            {
                "registered_model_name": "house-price-regressor",
                "model_version": "monitoring-demo",
                "production_alias": "production",
                "run_id": "monitoring-demo-run",
                "model_gate_status": "PASS",
            },
        ),
        settings_loader=load_config,
    )
    with TestClient(app) as client:
        before = client.get("/metrics")
        assert before.status_code == 200
        before_text = before.text
        before_requests = metric_value(before_text, "mlops_http_requests_total{method=\"POST\",path=\"/predict\",status=\"200\"}") if False else None

        for area in (900, 1000, 1200):
            payload = {"instances": [{**INSTANCE, "Area_sqft": area}]}
            response = client.post("/predict", json=payload)
            assert response.status_code == 200, response.text

        after = client.get("/metrics")
        assert after.status_code == 200
        after_text = after.text
        assert "mlops_http_requests_total" in after_text
        assert "mlops_http_request_duration_seconds_bucket" in after_text
        assert "mlops_prediction_value_bucket" in after_text
        assert "mlops_prediction_requests_total 3.0" in after_text
        assert "mlops_prediction_errors_total 0.0" in after_text
        assert before_text != after_text
        # The /metrics scrape itself is also real HTTP traffic and appears in the request counter.
        assert 'mlops_http_requests_total{method="POST",path="/predict",status="200"} 3.0' in after_text
        print("PHASE13_SMOKE=PASS")
        print("real_predict_requests=3")
        print("prometheus_metrics_changed=true")
        print("prediction_requests_total=3")
        print("prediction_errors_total=0")
        selected = [
            line for line in after_text.splitlines()
            if line.startswith("mlops_http_requests_total{method=\"POST\",path=\"/predict\",status=\"200\"}")
            or line.startswith("mlops_prediction_requests_total ")
            or line.startswith("mlops_prediction_errors_total ")
            or line.startswith("mlops_prediction_value_count ")
            or line.startswith("mlops_prediction_value_sum ")
        ]
        evidence = Path("reports/phase13_evidence")
        evidence.mkdir(parents=True, exist_ok=True)
        (evidence / "monitoring_smoke_metrics.txt").write_text("\n".join(selected) + "\n", encoding="utf-8")
        print("prediction_distribution_observed=true")
        print("evidence_file=reports/phase13_evidence/monitoring_smoke_metrics.txt")


if __name__ == "__main__":
    main()
