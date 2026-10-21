# Project 4 — Phase 8 Handoff: Lineage Audit & Reproducibility

## Executive status

Phase 8 adds a read-only production lineage audit command and a repeat-run reproducibility check. It also strengthens Phase 7 so the registry cannot promote a model when its pipeline-created run, dataset identifier/hash, config path/hash, or real Git code version is missing.

**Important actual finding:** the bundled production model is a legacy baseline. Its `artifacts/models/production_model_metadata.json` has a matching local model SHA-256 and dataset hash, but it does **not** contain an MLflow registered model/version, MLflow run ID, configuration path/hash, or Git commit. The new audit correctly returns **FAIL** for this existing production model. No run ID, model version, or commit was fabricated, and the existing production model was not modified. A complete live lineage chain requires a genuine gate-approved promotion run from a Git checkout with MLflow available.

The supplied ZIP has no `.git` directory, so `git rev-parse HEAD` is unavailable in this extracted environment. The reproducibility report records this honestly and includes a deterministic SHA-256 fingerprint of the Python source tree as supplemental evidence; this fingerprint does not substitute for the required Git commit in production metadata.

## Required lineage chain

```text
Production Model
  └─ artifacts/models/production_model_metadata.json
       ├─ registered_model_name + model_version + production alias
       ├─ run_id / mlflow_run_id
       │    ├─ MLflow parameters (model, hyperparameters, preprocessing)
       │    ├─ MLflow metrics (MAE, RMSE, R2)
       │    └─ MLflow artifacts (model, configuration, validation, evaluation)
       ├─ dataset_identifier + dataset_sha256
       ├─ config_path + config_sha256
       └─ code_version (actual Git commit)
```

`verify-lineage` checks the local production artifact hash against metadata; required metadata fields; the configured MLflow production alias and version; version-to-run linkage; run tags/parameters for dataset, configuration and code lineage; MAE/RMSE/R2; and model/configuration/validation/evaluation artifacts. It writes `reports/lineage_verification.json`. The command is read-only and exits non-zero if any required link is missing or unverifiable.

## Reproducibility methodology

The new `reproducibility-check` command invokes the existing `run_training(settings, save=False)` twice with the same data, config, random seed, code, and runtime environment. It compares validation MAE, RMSE, and R² for both configured model roles. It does not write candidate models, change production, create MLflow runs, or change source data. The report records the data SHA-256, config SHA-256, Git version when available, and a stable SHA-256 over sorted Python source file paths and bytes. It checks the data and config hashes again after both runs to detect mutation.

### Tolerance justification

`config/config.yaml` sets `reproducibility.metric_tolerance: 1.0e-12`. The existing real-data reproducibility test already asserts exact equality for repeated same-environment runs. In the Phase 8 demonstration, both fresh runs produced exactly equal metrics for both models (observed absolute delta `0.0` for all six model/metric pairs). Therefore the configured `1e-12` is only a tiny floating-point comparison guard; it is not an accuracy allowance or a model-quality threshold. The actual observed deltas were zero.

## Actual verification results

### Reproducibility check

Command:

```bash
PYTHONPATH=src python -m mlops_pipeline reproducibility-check
```

Observed result: **PASS**. Dataset hash was `8696565fd2a80775efee9574463c81877a23cbf51314a72e5919c81936036a3c`; config hash was recorded in `reports/reproducibility_check.json`. All six metric comparisons passed, each with absolute delta `0.0` and tolerance `1e-12`. The dataset and config hashes remained unchanged.

### Lineage verification

Command:

```bash
PYTHONPATH=src python -m mlops_pipeline verify-lineage
```

Observed result: **FAIL, as expected for the legacy production metadata**. The local production artifact exists and its hash matches metadata, but the following required links are missing: `registered_model_name`, `model_version`, `run_id`, `config_path`, `config_sha256`, and `code_version`. The report states that the registry chain cannot be verified until a real gate-approved promotion records those links. MLflow is not installed in this environment, so no live registry request or UI verification is claimed.

### Automated tests

- `PYTHONPATH=src python -m pytest tests/unit -q` — **211 passed in 8.68s**.
- `PYTHONPATH=src python -m pytest tests/integration -q` — **32 passed in 27.21s**.
- Combined unit+integration run was attempted, but the command runner timed out after partial output. The two suites were then run separately and each completed successfully: **243 tests total passed across the two completed runs**.
- `PYTHONPATH=src python -m mlops_pipeline validate-config` — passed.

## Files changed

- `config/config.yaml` — adds configured reproducibility metric tolerance and rationale.
- `src/mlops_pipeline/config/schema.py` — validates `reproducibility.metric_tolerance`.
- `src/mlops_pipeline/monitoring/lineage.py` — read-only local + MLflow production lineage audit.
- `src/mlops_pipeline/monitoring/reproducibility.py` — two-run metric comparison, hashes, source-tree fingerprint, and report output.
- `src/mlops_pipeline/monitoring/mlflow_tracking.py` — persists dataset identifier and config path alongside run lineage in the model result.
- `src/mlops_pipeline/monitoring/model_registry.py` — refuses registry registration/promotion if required data/config/Git lineage is missing or the code version is only the unavailable placeholder; records dataset identifier and config path in production metadata.
- `src/mlops_pipeline/cli.py` — adds `verify-lineage` and `reproducibility-check` commands.
- `tests/unit/test_phase8_lineage_reproducibility.py` — lineage-gap, reproducibility-pass, and reproducibility-failure tests.
- `tests/unit/test_model_registry.py` — verifies incomplete code lineage blocks registry access.
- `PHASE_8_HANDOFF.md` — this document.

## Commands

From the repository root:

```bash
PYTHONPATH=src python -m mlops_pipeline validate-config
PYTHONPATH=src python -m mlops_pipeline verify-lineage
PYTHONPATH=src python -m mlops_pipeline reproducibility-check
PYTHONPATH=src python -m pytest tests/unit -q
PYTHONPATH=src python -m pytest tests/integration -q
```

For live registry verification, use a Git checkout with dependencies installed:

```bash
pip install -r requirements.txt
pip install -e . --no-deps
PYTHONPATH=src python -m mlops_pipeline orchestrate
PYTHONPATH=src python -m mlops_pipeline verify-lineage
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

The flow must pass both gates and register/promote the candidate through the pipeline. Do not manually create a run or edit production metadata to make the lineage audit pass.

## Evidence to capture

1. `reports/lineage_verification.json` showing every lineage check and any failed link.
2. `reports/reproducibility_check.json` showing hashes, source fingerprint, tolerance, six metric comparisons, and input immutability.
3. Terminal output for both commands and the test commands.
4. Once a real promotion has occurred: MLflow registered-model page with production alias/version; version details showing run ID; run parameters and metrics; artifact browser showing model/configuration/validation/evaluation; production metadata showing matching run/version/data/config/commit; and a fresh `verify-lineage` PASS report.
5. Do not present the current legacy metadata audit as a PASS; it correctly fails until genuine lineage is recorded.

## Known issues / limitations

- The current bundled production model predates MLflow Registry lineage, so its chain is incomplete and audit fails closed.
- MLflow is not installed in this environment; live backend lineage could not be inspected.
- The extracted ZIP does not include Git metadata. A real Git commit is required before the new promotion guard will permit production promotion; `GIT_COMMIT` may be supplied only when it identifies the actual source revision.
- The reproducibility check establishes same-environment repeatability. It does not prove bitwise-identical serialized model bytes across different OS/library versions.
- The lineage verifier is read-only; it does not synthesize or backfill historical run/version IDs.

## Phase 9 instructions

1. Start from a Git checkout and install the declared dependencies, including MLflow and Prefect.
2. Run the full pipeline so the run is created by training, the Model Gate passes, and Phase 7 registers/promotes the model.
3. Run `verify-lineage` and resolve any failed links without manually inventing metadata.
4. Add an operational lineage reconciliation/repair report for interrupted promotions, but never rewrite historical IDs or hashes.
5. Add cross-environment reproducibility evidence with pinned dependency versions and compare metrics using a tolerance justified by repeated measurements across supported environments.
6. Keep both Data Gate and Model Gate mandatory; do not bypass gate-controlled registry transitions.
