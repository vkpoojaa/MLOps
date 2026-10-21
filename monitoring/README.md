# Phase 13 Operational Monitoring

This stack is intentionally separate from data-drift and model-performance monitoring.
It observes serving behavior only:

- request count/rate;
- request latency;
- HTTP error rate;
- prediction-value distribution.

Prometheus scrapes `GET /metrics` from the API every 5 seconds. Grafana is provisioned with the Prometheus datasource and the `House Price API — Operational Monitoring` dashboard.

## Run

From the repository root:

```bash
export GIT_COMMIT="$(git rev-parse HEAD)"
export APP_VERSION="0.1.0"
docker compose -f monitoring/docker-compose.yml up --build
```

The API must have access to the configured MLflow production registry. Then generate real traffic, for example:

```bash
curl -X POST http://localhost:8000/predict \
  -H 'content-type: application/json' \
  -d '{"instances":[{"Area_sqft":1000,"Bedrooms":2,"Bathrooms":1,"Age_Years":10,"Distance_to_Center_km":5.5,"Neighborhood_Score":7.1,"Has_Parking":1,"Property_Condition":"Good","City_Zone":"South"}]}'
```

Open Grafana at `http://localhost:3000` and the provisioned dashboard. Prometheus is at `http://localhost:9090`.

## Metrics

| Metric | Meaning | Source | Grafana use |
|---|---|---|---|
| `mlops_http_requests_total` | Count of HTTP requests by method/path/status | FastAPI middleware | request rate and error rate |
| `mlops_http_request_duration_seconds` | HTTP request latency histogram | FastAPI middleware | p95 prediction latency |
| `mlops_prediction_requests_total` | Successful `/predict` requests that reached inference | `/predict` handler | total predictions |
| `mlops_prediction_errors_total` | Validation/inference errors in prediction handling | `/predict` handler | prediction error count |
| `mlops_prediction_value` | Histogram of predicted target values | successful predictions | prediction distribution |

These are operational metrics. No drift statistic, ground-truth performance metric, or training-data comparison is emitted here.

## Reproduce and verify

The repository includes `scripts/phase13_monitoring_smoke.py`. It starts the FastAPI app with a deterministic approved-model test double, sends real HTTP requests through the ASGI application, and verifies that the Prometheus scrape changes after traffic. This validates the application-to-Prometheus metric contract without fabricating a live Grafana result.

A live Grafana/Prometheus acceptance run requires a Docker-capable environment. This execution environment did not provide Docker, so no claim is made that a live Grafana panel was observed here.


## Phase 14 ML/data monitoring

Phase 14 is separate from these operational metrics. Run the supplied incoming-batch monitor with:

```bash
PYTHONPATH=src python -m mlops_pipeline monitor-batches
```

It compares all three supplied 500-row batches with the training reference using empirically calibrated PSI thresholds and evaluates actual `Price_Lakhs` with the production model. See `PHASE_14_HANDOFF.md` and `reports/phase14_evidence/` for methodology and evidence.
