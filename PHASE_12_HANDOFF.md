# PHASE 12 HANDOFF — Registry-Based Production Rollback

## 1. Objective

Phase 12 implements and tests rollback to the previous known-good production model version using the existing MLflow Model Registry version/alias mechanism.

The rollback is **registry-only**:

- no model file is manually copied;
- no model file is downloaded back into the repository;
- the configured `production` alias is moved to the target registered version;
- the FastAPI serving layer resolves that alias and then loads the exact resolved version.

This matches the project's Phase 7 alias-based registry architecture and keeps the production transition inside the Prefect pipeline.

## 2. Implemented changes

### `src/mlops_pipeline/monitoring/model_registry.py`

Added `rollback_production_to_version(...)`.

Safety and correctness behavior:

- requires MLflow tracking and Model Registry to be enabled;
- resolves the current `production` alias before changing anything;
- supports `expected_current_version` to prevent a stale rollback from overwriting an unexpected newer version;
- requires the target registered version to exist;
- requires `model_gate.status=PASS` on the target;
- switches the MLflow `production` alias to the target version;
- re-resolves the alias and verifies the final version;
- restores the previous alias if post-transition metadata work fails;
- updates production metadata from registry/run lineage without touching the model file;
- records `model_file_copied=false`.

Promotion now also stores registry tags needed for later rollback lineage, including:

- `model_name`
- `artifact.sha256`
- `promoted_at_utc`

### `src/mlops_pipeline/pipeline/prefect_flow.py`

Added:

- `rollback_production_task`
- `rollback_production_flow`

The rollback transition is therefore executed as a named Prefect pipeline task (`9-rollback-production`) rather than by ad-hoc file manipulation.

### `src/mlops_pipeline/cli.py`

Added:

```text
rollback-production <target_version>
```

with optional:

- `--expected-current-version`
- `--reason`
- `--config`

The command delegates to the Prefect rollback flow.

### `src/mlops_pipeline/monitoring/lineage.py`

Lineage verification now understands the registry-authoritative state created by rollback. It does not falsely require the stale local joblib mirror to be rewritten after a registry-only rollback.

### `tests/unit/test_phase12_rollback.py`

Added five tests covering:

1. Production v1 → v2 promotion → rollback → v1.
2. Rollback safety when the expected current version does not match.
3. Serving loader resolving the rolled-back production alias to the exact version.
4. API health exposing the rolled-back model version.
5. Rollback being exposed as a Prefect pipeline task/flow.

## 3. Required demonstration

The controlled test demonstrates:

```text
Production v1
   ↓
v2 successfully promoted
   ↓
production alias = v2
   ↓
rollback target = v1
   ↓
production alias = v1
   ↓
serving loader = models:/house-price-regressor/1
```

The test records and asserts:

- before rollback production version = `2`
- promoted/current version being rolled back = `2`
- rollback action = `set_production_alias`
- final production version = `1`
- model file copied = `false`

It also proves that the local v2 model mirror remains byte-for-byte unchanged during rollback. That is intentional: the registry alias is authoritative for serving.

## 4. Rollback audit artifacts

Every successful rollback writes:

- `reports/rollback_history.jsonl` — append-only rollback events;
- `reports/rollback_latest.json` — latest rollback event.

Each event contains:

- `before_production_version`
- `promoted_version`
- `rollback_target_version`
- `rollback_action`
- `final_production_version`
- `reason`
- `target_run_id`
- `model_file_copied`

## 5. API/deployment behavior

`src/mlops_pipeline/serving/registry_loader.py` already resolves the configured production alias first and then loads the exact resolved version URI:

```text
models:/house-price-regressor/<resolved-version>
```

There is no local joblib fallback.

Therefore, after a registry rollback, restarting the API causes the deployment layer to resolve the rolled-back version. The Phase 12 tests verify that the loader resolves version `1` after the rollback sequence and that `/health` reports the rolled-back model version.

## 6. Verification results

Executed locally:

| Check | Result |
|---|---:|
| Phase 12 rollback tests | **5 passed** |
| Full unit suite | **231 passed** |
| Full integration suite | **32 passed** |
| Config validation | **PASS** |
| Python compileall | **PASS** |

Evidence files:

- `reports/phase12_evidence/rollback_test.log`
- `reports/phase12_evidence/config_validation.log`
- `reports/phase12_evidence/unit_tests.log`
- `reports/phase12_evidence/integration_tests.log`
- `reports/phase12_evidence/compileall.log`

## 7. Honest evidence boundary

The Phase 12 rollback test uses a deterministic registry test double because MLflow is not installed in the supplied execution environment and package download is unavailable. The test exercises the same registry-facing functions and alias/version contract as the live MLflow implementation.

No claim is made that a live MLflow server, GitHub Actions run, GHCR image, or remote production container was rolled back in this environment.

For the final deployment acceptance, run the same flow against the real MLflow registry and capture:

1. v1 as current production.
2. v2 promotion through the normal pipeline.
3. rollback to v1 through `rollback-production`.
4. MLflow alias/version evidence before and after.
5. restarted API `/health` showing the rolled-back version.
6. rollback audit files and deployment logs.

## 8. Suggested live command

From the repository root, after the real MLflow registry is configured:

```bash
PYTHONPATH=src python -m mlops_pipeline rollback-production 1 \
  --expected-current-version 2 \
  --reason "restore previous known-good production version"
```

Do not manually copy model artifacts during rollback.
