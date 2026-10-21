# Project 4 — Phase 7 Handoff: MLflow Model Registry

## Status and verification boundary

Phase 7 adds pipeline-controlled model registration and promotion after the Phase 6 Model Quality Gate. It preserves the existing model algorithms, feature engineering, preprocessing, split, evaluation methodology, and thresholds.

- Automated suite: **239 passed** (`PYTHONPATH=src python -m pytest -q`; the test runner printed `239 passed in 33.54s`).
- Configuration validation: passed (`PYTHONPATH=src python -m mlops_pipeline validate-config`).
- Registry behavior: verified with a deterministic fake MLflow client, including approved promotion, run/version linkage, rejection before any registry call, and local production-artifact rollback when the production alias transition raises an error.
- Live MLflow registry/UI verification: **not completed**. MLflow is not installed in this environment, and Prefect is also unavailable. The real pipeline was attempted and exited before fitting with the explicit error that MLflow tracking is enabled but MLflow is not installed. No real backend run ID or live registry version is claimed.

## Architecture and flow diagram

```text
Ingestion
   ↓
Validation
   ↓
Data Gate ── FAIL → stop; no training run, registry registration, or promotion
   ↓ PASS
Feature Engineering
   ↓
Training (pipeline creates MLflow runs before fit)
   ↓
Evaluation (metrics + artifacts)
   ↓
Model Quality Gate
   ├── FAIL → reject candidate; registry task is not invoked; production unchanged
   └── PASS
         ↓
      Register candidate from the existing MLflow run artifact
         ↓
      Set configured `candidate` alias on new model version
         ↓
      Prepare local production mirror and metadata with rollback snapshots
         ↓
      Set configured `production` alias as the final production transition
         ↓
      Persist registry/version/run lineage in reports and metadata
```

`registry_promotion_task` is the only pipeline task that invokes registry transitions. It is conditionally called only when `model_gate.passed == true`. The registration helper independently checks the gate decision too, so a direct call with a failed decision returns without loading MLflow or registering anything.

## Registry structure

Configured in `config/config.yaml`:

```yaml
model_registry:
  enabled: true
  registered_model_name: house-price-regressor
  candidate_alias: candidate
  production_alias: production
  await_registration_for_seconds: 120
```

The project pins MLflow in `requirements.txt` to `mlflow>=2.12,<4`. Promotion uses model-version **aliases** (`set_registered_model_alias`) rather than deprecated model stages such as `Production`/`Staging`.

- Registered model: `house-price-regressor`
- Candidate alias: `candidate`
- Production alias: `production`
- Candidate model URI: `runs:/<pipeline-created-run-id>/phase5/model`
- New model version is registered from the candidate's already tracked training run; Phase 7 does not create a second run.

## Production model identification

The configured MLflow `production` alias is the registry-level identifier for the promoted model version. Each promoted version is linked to the MLflow run ID through its source URI and version tags. The existing local `artifacts/models/selected_model.joblib` and `artifacts/models/production_model_metadata.json` remain compatibility mirrors used by the Phase 6 quality comparison and other local consumers.

The metadata records:

- registered model name and version
- configured production/candidate aliases
- pipeline-created MLflow run ID
- candidate validation metrics and primary metric
- dataset/configuration hashes and code version
- local production artifact path and SHA-256
- gate status and UTC promotion timestamp

The repository's pre-registry production model is initially identified through the Phase 6 local production metadata/artifact. On the first approved Phase 7 promotion, the pipeline creates a registered version and points the `production` alias at it.

## Promotion logic and safety

1. The flow evaluates the candidate using the existing configured primary metric and Phase 6 checks.
2. On Model Gate failure, it does not call the registry task. The helper also refuses a failed gate if called directly.
3. On pass, it requires the candidate's pipeline-created MLflow run ID and logged model artifact.
4. It registers a model version from `runs:/<run_id>/phase5/model`, sets the candidate alias, and tags the version with gate status, run ID, dataset/config hashes, and code version.
5. It prepares the local production model and metadata using atomic writes and snapshots the old bytes.
6. It sets the production alias as the final registry transition. If that transition raises an error, the local model and metadata are restored byte-for-byte and the exception propagates; the prior production alias is not intentionally changed by the failing call.
7. On success, the gate/evaluation reports record registration, model version, run ID, aliases, and artifact hash.

Registration is not the same as promotion: a registered candidate can exist under the `candidate` alias without becoming production. No Model Gate failure can create a candidate registration through this pipeline path.

## Before/after evidence

### Approved candidate — automated fake-client test

Test: `tests/unit/test_model_registry.py::test_approved_candidate_is_registered_linked_and_promoted`

- Existing local production bytes start as `old-production-pipeline`.
- The fake MLflow registry returns model version `7` for a candidate linked to run ID `pipeline-run-123`.
- The helper sets aliases `candidate` and `production` to version `7`.
- The local production mirror changes to the candidate pipeline bytes only after the approved promotion path.
- Production metadata records version `7`, run ID `pipeline-run-123`, hashes, metrics, and aliases.
- The evaluation report records `model_registry.status == PROMOTED`.

These are deterministic API-contract tests, not evidence of a real MLflow server run.

### Rejected candidate — gate blocks registry

Test: `tests/integration/test_prefect_flow.py::test_poor_candidate_is_rejected_and_real_production_artifact_is_preserved`

- A controlled poor candidate is given MAE `50.0` as a test-only gate input.
- The Model Gate fails and the registry task is patched to fail the test if invoked.
- Production artifact SHA-256 before and after is identical.
- The summary reports `REJECTED_BY_MODEL_GATE`; registration and promotion are false.

### Registry promotion operation fails

Test: `tests/unit/test_model_registry.py::test_registry_alias_failure_restores_existing_production_files`

- The fake registry allows candidate alias assignment but raises during the production alias transition.
- Existing production model bytes and metadata bytes are restored exactly.
- The production alias operation is recorded as failed.

## Commands

From the repository root, in an environment with network/package access:

```bash
pip install -r requirements.txt
pip install -e . --no-deps
PYTHONPATH=src python -m mlops_pipeline validate-config
PYTHONPATH=src python -m pytest -q
PYTHONPATH=src python -m mlops_pipeline orchestrate
```

Start the MLflow UI against the configured backend:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Open `http://127.0.0.1:5000`, select the `house-price-regression` experiment, open the registered model `house-price-regressor`, and inspect its versions and aliases. If the tracking URI resolves to an absolute database path, pass that same absolute path to the UI command.

## Live evidence / screenshot instructions

After running with real MLflow and Prefect installed, capture:

1. Prefect flow view showing the Model Gate before the `8-register-and-promote` task.
2. MLflow experiment run details for the candidate, including the same run ID used by the registered model version.
3. Registered model page showing the candidate version and `candidate` alias.
4. Successful promotion evidence showing `production` alias pointing to the approved version.
5. `artifacts/models/production_model_metadata.json` showing registered model name, version, run ID, metrics, and artifact SHA-256.
6. Rejected candidate test output and before/after production SHA-256 hashes.
7. If testing a failed registry alias transition, capture the test result proving local production files were restored.

Do not create an MLflow run manually after training. Run the pipeline so the run is created before fitting and then registered from its logged artifact.

## Tests and actual outcomes

- `PYTHONPATH=src python -m mlops_pipeline validate-config` — passed.
- `PYTHONPATH=src python -m pytest tests/unit/test_model_registry.py tests/integration/test_prefect_flow.py -q` — **7 passed in 16.33s** at the first targeted run; subsequent full suite includes these tests.
- `PYTHONPATH=src python -m pytest -q` — **239 passed in 33.54s** (pytest printed the complete successful summary; the surrounding execution wrapper did not exit cleanly after the summary).
- Real `PYTHONPATH=src python -m mlops_pipeline orchestrate` — **blocked**, exit code 1: `RuntimeError: MLflow tracking is enabled in config.yaml but MLflow is not installed.` The CLI also warned Prefect is unavailable. No live registry/UI claim is made.

## Files added/changed

- `config/config.yaml` — added registry name, aliases, enable flag, and registration wait timeout.
- `src/mlops_pipeline/config/schema.py` — typed `ModelRegistrySection` and alias consistency validation.
- `src/mlops_pipeline/config/validation.py` — rejects registry enabled while MLflow tracking is disabled.
- `src/mlops_pipeline/monitoring/model_registry.py` — candidate registration, run/version lineage tags, alias promotion, local mirror metadata, and rollback behavior.
- `src/mlops_pipeline/pipeline/prefect_flow.py` — conditional `8-register-and-promote` task only after a passing Model Gate; rejected candidates never invoke it.
- `tests/unit/test_model_registry.py` — gate blocking, approved promotion, run/version lineage, and rollback tests.
- `tests/integration/test_prefect_flow.py` — rejected candidate blocks registry; passing gate reaches the promotion integration point.
- `PHASE_7_HANDOFF.md` — this document.

## Known issues / limitations

1. Live MLflow Registry and Prefect UI behavior remains unverified in this environment because neither package is installed and package installation is unavailable here.
2. Fake-client tests verify the MLflow API contract and transition ordering, not remote backend permissions, artifact-store reachability, or server-side behavior.
3. The production alias is switched only after the local mirror is staged. The local model/metadata are rolled back if the alias switch throws. A process crash at the exact boundary between local writes and alias update can still require operational reconciliation; inspect the registry alias and metadata before rerunning after an interrupted process.
4. Model Registry promotion is now implemented; Model Registry rollback automation, serving, deployment, and CI/CD are not included.
5. Model Gate thresholds and ML methodology remain unchanged.

## Phase 8 instructions

1. Inspect all handoffs and current registry/flow implementation first.
2. Install the configured dependencies and verify a real Prefect flow and MLflow Registry/UI run; save screenshots or command output as evidence.
3. Add an explicit registry consistency/reconciliation command that compares the production alias, registered model version, run ID, local production metadata, and local artifact hash before future promotion.
4. Add operational rollback to a prior registered version using aliases, with tests proving rollback does not create a new training run and records the selected version/run lineage.
5. Add a dry-run mode that evaluates the gates and reports intended registry actions without registering or promoting.
6. Keep Model Gate and Data Gate mandatory; no registration or promotion may bypass them. Do not add serving/deployment unless the Phase 8 scope explicitly requires it.
