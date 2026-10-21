# PHASE 13 HANDOFF — Prometheus + Grafana Operational Monitoring

## 1. Objective

Phase 13 adds operational monitoring for the containerised house-price prediction API using **Prometheus** and **Grafana**, as required by the official Project 4 brief.

The monitoring scope is deliberately limited to serving operations:

- request count / request rate;
- request latency;
- HTTP error rate;
- prediction-value distribution.

This is separate from data-drift and model-performance monitoring. No drift calculation or ground-truth model-quality metric is added to the operational metrics endpoint.

## 2. Official-brief alignment

The supplied capstone brief requires the serving layer to expose operational metrics covering **latency, throughput, errors and the distribution of predicted prices**, and requires a monitoring dashboard/report covering the incoming batches.

Phase 13 implements the operational part of that requirement with a Prometheus scrape endpoint and a provisioned Grafana dashboard.

## 3. Implemented changes

### API metrics

`src/mlops_pipeline/serving/api.py` now uses `prometheus-client` with an application-local `CollectorRegistry`.

`GET /metrics` is now a Prometheus text exposition endpoint.

The previous JSON diagnostic representation remains available at:

```text
GET /metrics/json
```

### Prometheus metrics

| Metric | Type | Meaning | Source |
|---|---|---|---|
| `mlops_http_requests_total` | Counter | Total HTTP requests, labelled by method/path/status | FastAPI HTTP middleware |
| `mlops_http_request_duration_seconds` | Histogram | HTTP request latency | FastAPI HTTP middleware |
| `mlops_prediction_requests_total` | Counter | Successful prediction requests that reached inference | `/predict` handler |
| `mlops_prediction_errors_total` | Counter | Prediction validation/inference errors | `/predict` handler |
| `mlops_prediction_value` | Histogram | Distribution of predicted house prices in configured target units | successful `/predict` responses |

### Prometheus

Added:

```text
monitoring/prometheus/prometheus.yml
```

Prometheus scrapes:

```text
api:8000/metrics
```

at a 5-second interval.

### Grafana

Added:

```text
monitoring/grafana/provisioning/datasources/datasource.yml
monitoring/grafana/provisioning/dashboards/dashboard.yml
monitoring/grafana/dashboards/house-price-operational.json
```

The dashboard is provisioned automatically and contains panels for:

1. request rate;
2. p95 request latency;
3. HTTP error rate;
4. prediction distribution;
5. total predictions;
6. prediction error count;
7. Prometheus scrape/API availability.

### Reproducible local stack

Added:

```text
monitoring/docker-compose.yml
monitoring/README.md
```

The stack provides the API, Prometheus and Grafana services. The API continues to use the existing MLflow production-registry architecture; it does not load an arbitrary local model.

## 4. Real traffic verification

Added:

```text
scripts/phase13_monitoring_smoke.py
```

This test creates the real FastAPI application, sends real HTTP requests through `TestClient`, scrapes the actual `/metrics` endpoint, and verifies that the Prometheus series change.

Executed result:

```text
PHASE13_SMOKE=PASS
real_predict_requests=3
prometheus_metrics_changed=true
prediction_requests_total=3
prediction_errors_total=0
prediction_distribution_observed=true
```

The captured Prometheus values are stored in:

```text
reports/phase13_evidence/monitoring_smoke_metrics.txt
```

Observed values from that run:

```text
mlops_http_requests_total{method="POST",path="/predict",status="200"} 3.0
mlops_prediction_value_count 3.0
mlops_prediction_value_sum 3400.0
mlops_prediction_requests_total 3.0
mlops_prediction_errors_total 0.0
```

These are actual values produced by the test traffic, not fabricated Grafana values.

## 5. Grafana verification boundary

The dashboard JSON, Prometheus configuration and provisioning files were parsed and validated locally.

The execution environment used for this phase does **not** provide Docker or Docker Compose. Therefore a live Prometheus server and Grafana instance could not be started here, and no claim is made that a live Grafana panel was visually observed or that a live dashboard time series changed in this environment.

The implementation is designed so that the same real API traffic updates Prometheus, and Grafana reads those Prometheus series through the provisioned datasource.

For final live acceptance in a Docker-capable environment, run:

```bash
docker compose -f monitoring/docker-compose.yml up --build
```

Then generate prediction traffic and open Grafana on port `3000`. The provisioned dashboard is:

```text
House Price API — Operational Monitoring
```

Prometheus is exposed on port `9090`.

## 6. Reproduction

### Start the monitoring stack

```bash
docker compose -f monitoring/docker-compose.yml up --build
```

The API must have access to the configured MLflow tracking/registry backend and an approved `production` model alias.

### Generate API traffic

Example:

```bash
curl -X POST http://localhost:8000/predict \
  -H 'content-type: application/json' \
  -d '{"instances":[{"Area_sqft":1000,"Bedrooms":2,"Bathrooms":1,"Age_Years":10,"Distance_to_Center_km":5.5,"Neighborhood_Score":7.1,"Has_Parking":1,"Property_Condition":"Good","City_Zone":"South"}]}'
```

Send multiple requests with different valid feature values to change request-rate, latency and prediction-distribution panels. Send a deliberately invalid request to exercise the error-rate panel.

### Prometheus

Open the Prometheus UI and query, for example:

```promql
sum(rate(mlops_http_requests_total{path="/predict"}[$__rate_interval]))
```

```promql
histogram_quantile(0.95, sum by (le) (rate(mlops_http_request_duration_seconds_bucket{path="/predict"}[$__rate_interval])))
```

```promql
100 * sum(rate(mlops_http_requests_total{path="/predict",status=~"4..|5.."}[$__rate_interval])) / clamp_min(sum(rate(mlops_http_requests_total{path="/predict"}[$__rate_interval])), 1e-9)
```

```promql
sum(rate(mlops_prediction_value_bucket[$__rate_interval])) by (le)
```

### Grafana

Grafana automatically receives the Prometheus datasource and dashboard through the provisioning files. No manual dashboard import is required.

## 7. Tests

Phase-specific tests:

```text
10 passed
```

Full unit suite:

```text
234 passed
```

Full integration suite:

```text
32 passed
```

Therefore the combined existing test suites contain:

```text
266 passed
```

Additional checks:

- config validation: **PASS**
- Prometheus/Grafana YAML parsing: **PASS**
- Grafana dashboard JSON parsing: **PASS**
- Python compileall: **PASS**
- real API traffic → `/metrics` change: **PASS**

## 8. Separation from drift/performance monitoring

Operational monitoring is intentionally not mixed with the later drift/performance monitoring layer.

Operational metrics answer questions such as:

- Are requests reaching the service?
- How many requests are arriving?
- How long do requests take?
- How many requests fail?
- What values is the service predicting?

They do **not** answer:

- Has the incoming feature distribution drifted?
- Has prediction accuracy degraded against ground truth?
- Has the model become statistically stale?

Those concerns remain separate from the Prometheus/Grafana serving dashboard.

## 9. Files added/changed

```text
src/mlops_pipeline/serving/api.py
requirements.txt
requirements-serving.lock
monitoring/prometheus/prometheus.yml
monitoring/grafana/provisioning/datasources/datasource.yml
monitoring/grafana/provisioning/dashboards/dashboard.yml
monitoring/grafana/dashboards/house-price-operational.json
monitoring/docker-compose.yml
monitoring/README.md
scripts/phase13_monitoring_smoke.py
tests/unit/test_phase13_monitoring.py
reports/phase13_evidence/monitoring_smoke.log
reports/phase13_evidence/monitoring_smoke_metrics.txt
PHASE_13_HANDOFF.md
```
