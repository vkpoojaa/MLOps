# Phase 9 Handoff — Registry-Backed Prediction API

## Outcome

Phase 9 adds a FastAPI deployment layer for the house-price pipeline. The API is intentionally fail-closed: it resolves the configured **MLflow Model Registry production alias** established in Phase 7 and loads the exact registered version resolved by that alias. It does not load `selected_model.joblib`, a candidate artifact, or any arbitrary local model file.

## Endpoints

| Method | Endpoint | Behavior |
|---|---|---|
| `GET` | `/health` | Returns `200` and deployed model metadata when the approved model is loaded; otherwise `503` with a safe availability message. |
| `POST` | `/predict` | Accepts a batch of 1–100 instances, validates the configured feature schema, predicts, and returns predictions plus model/version metadata. |
| `GET` | `/metrics` | Returns request/error counts and accumulated/mean prediction latency in JSON. |
| `GET` | `/docs` | FastAPI interactive API documentation. |

### Prediction payload

Send `instances` as a list of feature objects. Each object must contain exactly the configured feature columns. Numeric values must be finite numbers or `null`; categorical values must be strings or `null`. Extra or missing feature names and invalid types return `422`. The target and excluded identifier/batch columns are not accepted as prediction features.

```json
{
  "instances": [
    {
      "Area_sqft": 1000,
      "Bedrooms": 2,
      "Bathrooms": 1,
      "Age_Years": 10,
      "Distance_to_Center_km": 5.5,
      "Neighborhood_Score": 7.1,
      "Has_Parking": 1,
      "Property_Condition": "Good",
      "City_Zone": "South"
    }
  ]
}
```

A successful response includes `predictions`, `prediction_unit` (read from the configured target), `count`, and `model` metadata including registered model name, resolved version, production alias, MLflow run ID, gate status, and lineage tags when present.

## Registry approval and model loading

1. The API reads the validated `config/config.yaml`.
2. It uses the configured registry model name and production alias (currently `house-price-regressor@production`).
3. It resolves that alias using `MlflowClient.get_model_version_by_alias`.
4. It refuses to load a version unless its registry tag `model_gate.status` is exactly `PASS` and the version/run lineage is present.
5. It loads the exact resolved registered version through MLflow's PyFunc interface. The exact version is used after alias resolution so a concurrent alias change cannot cause the response metadata and loaded artifact to disagree.
6. The logged scikit-learn pipeline performs its original imputation, encoding, scaling, and prediction. FastAPI only checks the request schema and arranges columns; it does not duplicate preprocessing.

If configuration, MLflow, the alias, the approval tag, lineage, or model artifact cannot be resolved, startup records the failure and the service stays available for diagnostics. `/health` and `/predict` return `503`; there is no local-artifact fallback. Restart the service after registry/MLflow availability is restored to retry model loading.

## Run locally

Install the project dependencies from the repository root:

```bash
python -m pip install -r requirements.txt
```

Start the API:

```bash
PYTHONPATH=src python -m mlops_pipeline.serving
```

Alternatively:

```bash
PYTHONPATH=src uvicorn mlops_pipeline.serving.api:app --host 127.0.0.1 --port 8000
```

Then open `http://127.0.0.1:8000/docs`, or call `/health`, `/predict`, and `/metrics`.

The API process must have access to the same MLflow tracking/registry backend and artifacts used for Phase 7 promotion. For the current SQLite tracking URI, that means the correct `mlflow.db` and artifact store must be available at the configured location. Do not copy a candidate model into the local models directory as a workaround.

## Files added/changed

- `src/mlops_pipeline/serving/api.py` — app factory, endpoints, request validation, safe errors, counters/latency.
- `src/mlops_pipeline/serving/registry_loader.py` — strict approved-alias resolution and registered-version loading.
- `src/mlops_pipeline/serving/__main__.py` — local Uvicorn entry point.
- `tests/unit/test_phase9_api.py` — endpoint, schema, metadata, metrics, load-failure, alias-resolution, and approval-tag tests.
- `requirements.txt` — FastAPI, Uvicorn, and HTTPX runtime/test dependencies.

## Verification performed

- `PYTHONPATH=src pytest -q tests/unit`: **218 passed**.
- `PYTHONPATH=src pytest -q tests/integration`: **32 passed**.
- `python -m compileall -q src tests`: passed.
- `PYTHONPATH=src python -m mlops_pipeline validate-config`: configuration valid.
- API tests cover health metadata, valid prediction, missing feature, malformed top-level request, metrics, model-loading failure behavior, production-alias resolution, exact-version loading, and rejection of a non-approved version.
- Started the real Uvicorn app locally and called all three endpoints. `/metrics` returned `200`; `/health` and `/predict` correctly returned `503` because this execution environment does not have MLflow installed and cannot resolve the approved production alias. No local model was substituted.

### Important environment limitation

A **real registry-backed production prediction has not been verified in this environment**. MLflow is not installed here, and the bundled Phase 8 production metadata was already identified as legacy/incomplete; it is not proof of a currently approved registry version. Therefore, the local live server correctly fails closed. The valid-prediction API test uses an injected deterministic model double to exercise the endpoint contract and response path; it does not claim to be a prediction from the real production artifact. After installing dependencies and connecting to the Phase 7 registry containing a gate-approved `production` alias, restart the API and confirm `/health` reports the actual registered version before sending production traffic.

## Test commands

```bash
PYTHONPATH=src pytest -q tests/unit/test_phase9_api.py
PYTHONPATH=src pytest -q tests/unit
PYTHONPATH=src pytest -q tests/integration
PYTHONPATH=src python -m mlops_pipeline validate-config
```

## Deployment cautions / next phase

- Keep the MLflow tracking URI, registry database, and artifact store durable and accessible to the serving process.
- Protect the API with authentication, TLS, rate limits, and network policy before public exposure; these are not implemented in this phase.
- Add a deployment health/readiness probe using `/health`; do not mark the service ready on `503`.
- Add persistent metrics/monitoring and controlled model refresh/restart behavior before production operations.
