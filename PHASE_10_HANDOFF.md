# Phase 10 Handoff — Containerized Registry-Backed Prediction API

## Status summary

Phase 10 adds a Docker image definition and a Git-aware image build script for the Phase 9 FastAPI service. The container intentionally contains application code and configuration only. It does **not** contain `artifacts/`, `data/`, a local `mlflow.db`, MLflow run directories, or a copied model. At startup the existing Phase 9 loader must resolve the configured MLflow Model Registry `production` alias, verify `model_gate.status == PASS`, and load the exact approved registered version. If that process fails, `/health` and `/predict` fail closed; there is no local model fallback.

**Important verification boundary:** this execution environment has no Docker/Podman/Buildah engine, no `.git` directory in the supplied project ZIP, and no installed MLflow client. The Docker build and container-level endpoint checks therefore could not be run here. A real host Uvicorn process was started and endpoint behavior was checked. The service correctly refused prediction when the production registry could not be resolved. No successful real-registry prediction is claimed.

## Files added / changed

- `Dockerfile` — Python 3.12.3 slim Bookworm base, exact direct serving dependency pins, non-root runtime user, build/version labels, healthcheck, port 8000, and API entrypoint.
- `.dockerignore` — excludes Git history, local environments, secrets, training data, model files, local registry state, reports, tests, notebooks, and docs from the build context.
- `requirements-serving.lock` — exact versions for direct runtime dependencies only: FastAPI, Uvicorn, Pydantic, PyYAML, NumPy, pandas, scikit-learn, joblib, and MLflow. Prefect, pytest, and development extras are omitted.
- `scripts/build-image.sh` — requires a real Git checkout, obtains the full and short commit SHA from Git, and builds an image tagged `<image>:<version>-<short-commit>` with OCI version and revision labels.
- `src/mlops_pipeline/serving/__main__.py` — binds to `0.0.0.0` inside the container and supports `HOST`/`PORT` environment overrides.
- `src/mlops_pipeline/serving/api.py` — `/health` and `/metrics` now include image/version/commit metadata from `APP_VERSION`, `GIT_COMMIT`, and `IMAGE_NAME`.
- `tests/unit/test_phase10_container.py` — container hardening, traceability, no-model-copy, dependency-scope, and build metadata tests.
- `reports/phase10_container_verification.json` — machine-readable record of checks and blockers.

## Build and Git provenance

The Dockerfile has OCI labels:

- `org.opencontainers.image.version`
- `org.opencontainers.image.revision`

The same version, commit SHA, and image name are exposed in `/health` and `/metrics` as `build.version`, `build.git_commit`, and `build.image`.

Build from the real repository checkout:

```bash
./scripts/build-image.sh
```

The script reads `git rev-parse HEAD`, uses the short SHA in the image tag, and passes the full SHA as `GIT_COMMIT`. The Dockerfile intentionally fails if `GIT_COMMIT` is absent or `unknown`; do not invent a commit for a source ZIP. Inspect the image labels after building:

```bash
docker image inspect mlops-house-price-api:0.1.0-<short-commit> \
  --format '{{json .Config.Labels}}'
```

For a manual build from a Git checkout, supply all provenance arguments explicitly:

```bash
docker build \
  --build-arg APP_VERSION=0.1.0 \
  --build-arg GIT_COMMIT="$(git rev-parse HEAD)" \
  --build-arg IMAGE_NAME=mlops-house-price-api \
  -t "mlops-house-price-api:0.1.0-$(git rev-parse --short=12 HEAD)" .
```

### Reproducibility note

The Python base image tag and direct application dependencies are version-pinned, and the build does not depend on a local virtual environment or model artifact. However, `requirements-serving.lock` currently pins **direct** runtime packages, not every transitive dependency with hashes. A fully hermetic/supply-chain reproducible build still needs a generated transitive lock (ideally hash-locked) and a digest-pinned base image. This environment had no package-index network access, so I could not resolve and verify that full lock honestly. Do not treat this as a fully hash-reproducible image yet.

## Run the container

The image does not bundle MLflow state or models. Configure it to reach the same registry and artifact store used by Phase 7. For deployment, provide a read-only config file whose `mlflow.tracking_uri` points to the real registry backend and whose `model_registry.registered_model_name` and `production_alias` match the approved deployment. Do not mount or copy a local candidate model as a workaround.

```bash
docker run --rm --name house-price-api \
  -p 8000:8000 \
  -e MLOPS_CONFIG=/run/config/config.yaml \
  -v "$PWD/config/deployment.yaml:/run/config/config.yaml:ro" \
  mlops-house-price-api:0.1.0-<short-commit>
```

Use the actual version/commit tag printed by the build script. Supply MLflow credentials and any artifact-store credentials through the deployment platform's secret mechanism, not a baked-in `.env` or Docker layer. If the approved alias cannot be loaded, the process remains available for diagnostics but the container healthcheck will fail because `/health` returns `503`.

No `docker-compose.yml` was added: the API should connect to the environment's existing MLflow tracking/registry and artifact services; inventing a local registry service or mounting an unapproved model would obscure the intended production architecture.

## Verification performed in this environment

| Check | Result | Evidence |
|---|---|---|
| Phase 9 API + Phase 10 container tests | **12 passed** | Test command below |
| Full unit suite | **223 passed** | `PYTHONPATH=src pytest -q tests/unit` |
| Integration suite | **32 passed** | `PYTHONPATH=src pytest -q tests/integration` |
| Config validation | Passed | `python -m mlops_pipeline validate-config` |
| Python compilation | Passed | `python -m compileall -q src tests` |
| Build script shell syntax | Passed | `sh -n scripts/build-image.sh` |
| Host Uvicorn process | Started | Bound to `0.0.0.0:18000` |
| Live `GET /health` | `503` as expected | MLflow is unavailable; build metadata returned |
| Live `GET /metrics` | `200` | JSON metrics and build metadata returned |
| Live invalid `POST /predict` | `422` | Malformed request rejected by FastAPI schema validation |
| Live valid-shaped `POST /predict` | `503` as expected | No registry-backed production model could be loaded; no local fallback |
| Docker image build / container start | **Not run: blocked** | `docker: command not found` |

The existing Phase 9 API tests exercise a successful prediction response using a deterministic injected model double and separately test registry alias resolution and refusal to load an unapproved model. That is an API contract test, **not** evidence of a successful prediction from a real production registry model.

Commands to repeat tests:

```bash
PYTHONPATH=src pytest -q tests/unit/test_phase9_api.py tests/unit/test_phase10_container.py
PYTHONPATH=src pytest -q tests/unit
PYTHONPATH=src pytest -q tests/integration
PYTHONPATH=src python -m mlops_pipeline validate-config
python -m compileall -q src tests
sh -n scripts/build-image.sh
```

## Deployment acceptance checklist

- [ ] Generate and verify a full transitive dependency lock with hashes.
- [ ] Build from a Git checkout; record the full commit and resulting image digest in CI/release metadata.
- [ ] Run the image against the real MLflow registry and artifact store.
- [ ] Confirm `/health` returns `200` and reports the approved registered model/version, alias `production`, gate status `PASS`, image version, and exact Git commit.
- [ ] Send a valid `/predict` request and verify the response model version matches `/health`.
- [ ] Verify invalid input returns `422`, `/metrics` returns `200`, and registry failure yields `/health`/`/predict` `503` with no local fallback.
- [ ] Store credentials as runtime secrets; use TLS/authentication/network controls before public exposure.
