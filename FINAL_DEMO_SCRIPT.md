# Final Demo Script — 5–10 Minutes

## Goal

Show the complete MLOps loop and one safe failure path without claiming unavailable live evidence. Keep the demo to 7–9 minutes.

## 0:00–0:45 — Architecture

Open `docs/ARCHITECTURE.md`.

Say:

> "The system is fail-closed. Data must pass the Data Gate before training. A candidate must pass the Model Gate before MLflow Registry promotion. FastAPI serves the registry production alias. Operational monitoring is separate from drift monitoring. Rollback changes the registry alias rather than copying model files."

Point out:

- YAML configuration source of truth;
- Prefect stage flow;
- MLflow/Registry;
- FastAPI → Prometheus → Grafana;
- incoming Batch 1/2/3 monitoring.

## 0:45–2:00 — Successful pipeline

Run:

```bash
PYTHONPATH=src python -m mlops_pipeline validate-config
PYTHONPATH=src python -m mlops_pipeline check-contract
PYTHONPATH=src python -m mlops_pipeline orchestrate
```

Show the successful stage/gate output and, if connected to the real environment, show the MLflow run/registered version. Do not invent a run ID or version.

Explain:

- ingestion;
- validation/Data Gate;
- preprocessing/training;
- evaluation;
- leakage checks;
- Model Gate.

## 2:00–3:00 — MLflow, lineage, and production model

If a live MLflow environment is available, show:

1. the run;
2. parameters/metrics;
3. registered model version;
4. production alias;
5. source/config/data lineage.

Run:

```bash
PYTHONPATH=src python -m mlops_pipeline verify-lineage
```

Do not claim live lineage if the command reports a missing/incomplete registry chain.

Then show the serving policy in `src/mlops_pipeline/serving/registry_loader.py`: the API resolves the approved production registry model rather than a local copied model file.

## 3:00–4:00 — API

Start FastAPI in the configured environment:

```bash
MLOPS_CONFIG=config/config.yaml PYTHONPATH=src uvicorn mlops_pipeline.serving.api:app --host 0.0.0.0 --port 8000
```

Show:

```text
GET /health
POST /predict
GET /metrics
```

Send one valid prediction and one invalid request. Show the HTTP response and the Prometheus metrics endpoint.

## 4:00–5:00 — Operational monitoring

If Docker is available:

```bash
export GIT_COMMIT="$(git rev-parse HEAD)"
docker compose -f monitoring/docker-compose.yml up --build
```

Open Prometheus/Grafana and send several predictions. Show the request rate, latency, errors, and prediction distribution panels changing.

If Docker/Grafana is unavailable, show the repository dashboard JSON and the recorded local Prometheus smoke evidence. Explicitly say that the local evidence proves API → Prometheus metric updates, not a live Grafana panel observation.

## 5:00–6:00 — Bad-data failure

Use the existing bad-data test/fixture rather than modifying the supplied CSV.

Show:

```text
invalid data
 -> validation/Data Gate failure
 -> training not invoked
 -> production unchanged
```

Say:

> "The important property is safe failure: the bad input does not reach training and cannot replace production."

## 6:00–7:00 — Bad-model failure

Show the Model Gate test for a poor candidate.

Then show the production-comparison case where a candidate passes the absolute threshold but is worse than production.

Explain:

> "An acceptable absolute score is not enough. The candidate must also be better than the current production model."

## 7:00–8:00 — Rollback

If live MLflow is available, show:

```bash
PYTHONPATH=src python -m mlops_pipeline rollback-production <previous_version> \
  --expected-current-version <current_version> \
  --reason "demo rollback"
```

Show the alias before and after, and show the API health model version after restart/reload.

If live MLflow is unavailable, show the Phase 12 registry-contract evidence and explain that the local test verifies the same registry alias/safety mechanism but is not live MLflow evidence.

## 8:00–9:00 — Drift results

Run or show:

```bash
PYTHONPATH=src python -m mlops_pipeline monitor-batches
```

Show the actual results:

| Batch | Feature drift | Performance concern | MAE |
|---|---|---|---:|
| 1 | No | No | 5.5694 |
| 2 | Yes | Yes | 6.8813 |
| 3 | Yes | Yes | 13.1237 |

Explain:

> "Drift is not model replacement. It is an investigation/retraining signal. Any new candidate still has to pass the Data Gate and Model Gate."

## 9:00–9:30 — Close

Show `FINAL_SUBMISSION_CHECKLIST.md`.

State clearly which items are live evidence and which require manual acceptance evidence. Do not claim a screenshot, GitHub run, release, Grafana panel, MLflow lineage, or team contribution record that has not actually been captured.

## Two-person split

### Member A

- architecture;
- data validation/Data Gate;
- training/evaluation/Model Gate;
- drift monitoring.

### Member B

- MLflow/Registry/lineage;
- FastAPI/Docker;
- Prometheus/Grafana;
- CI/CD and rollback.

Both members should be able to answer questions about the entire repository and make a small live modification if asked, consistent with the academic-integrity requirement in the official brief.
