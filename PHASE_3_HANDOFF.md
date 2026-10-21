# Phase 3 Handoff — Data Validation & Quality Gate

## 1. Status

Phase 3 implementation and verification are complete for the current repository snapshot. The reusable validator was present in the supplied Phase 3A archive; this phase completion integrated it into the training entry point, fixed a nullable-text dtype edge case exposed by the full suite, added fail-closed integration tests, and documented the result.

## 2. Architecture and execution point

Training path:

1. CLI loads and validates `config/config.yaml`.
2. `run_training(settings)` calls `load_training_data(settings)`; this reads only the configured historical training CSV and applies the existing `DataContract`.
3. `DataValidator.from_settings(settings).validate(df, "train")` runs the configurable quality checks.
4. `ValidationReport.raise_if_failed()` raises `DataValidationError` carrying the complete report if any check fails.
5. Only after the gate passes can the pipeline split data, fit/evaluate models, run leakage checks, or write model/evaluation artifacts.

The gate is in `src/mlops_pipeline/models/training.py`, immediately after data loading and before `split_train_validation`. A validation failure therefore occurs before any training or artifact-write operation. The CLI prints a readable validation report and exits with status 1. Direct Python callers receive `DataValidationError` and can inspect `.report` / `.report.to_dict()`.

## 3. Validation implementation and configuration

Reusable validator: `src/mlops_pipeline/data/validation.py`.

The validator does not mutate data or perform preprocessing. It returns a `ValidationReport` with overall `passed`, profile, row/column counts, status counts, and per-check results. Each check includes `name`, `status` (`pass`, `fail`, or `skipped`), optional `column`, `observed`, `expected`, and `reason`.

Rules are driven by `config/config.yaml` under `validation` and the existing `data_contract` column definitions. Current enabled check groups:

- `validation.checks.schema`: required/unexpected/duplicate columns, dtypes, minimum row counts.
- `validation.checks.missing_values`: per-column and overall missing fractions.
- `validation.checks.value_ranges`: numeric plausibility bounds and non-finite values.
- `validation.checks.categorical_values`: configured allowed values.
- `validation.checks.target`: target presence, missing/non-finite/range/constant checks.
- `validation.checks.duplicates`: duplicate records based on configured roles.
- `validation.checks.identifiers`: uniqueness, configured identifier pattern, and training-ID overlap where required.

Current notable configured thresholds (see `docs/DATA_VALIDATION.md` for the detailed rationale and observed-data context):

- Maximum missing fraction per nullable column: `Age_Years=0.08`, `Distance_to_Center_km=0.08`, `Neighborhood_Score=0.08`.
- Maximum overall missing-cell fraction: `0.015`.
- Numeric plausibility bounds: `Area_sqft` 100–6000; `Bedrooms` 1–10; `Bathrooms` 1–10; `Age_Years` 0–100; `Distance_to_Center_km` 0–100; `Price_Lakhs` 1–500.
- Target minimum distinct values: `2`.
- Maximum duplicate-record fraction: `0.0`, comparing configured `features` and `target` roles.
- Identifier pattern: `^HS-[0-9]{6}$`.
- Profiles define required/optional roles and minimum row counts: `train` (1000), `incoming` (100), `inference` (1). Inference requires features, not target or identifier.
- `validation.on_failure: halt` documents the fail-closed policy.

The existing config validation checks required values, types, known columns, and contradictory settings. No new runtime dependency was added. The current config loader reports nine intentionally pending settings for later model-gate/monitoring work; these do not disable the Phase 3 data gate.

## 4. Training versus inference

`train` requires the training identifier, target, and all features. `incoming` requires identifier, batch, target, and features, and checks disjointness from training identifiers when the caller supplies reference IDs. `inference` requires feature columns; target, identifier, and batch metadata are optional. Identical inference requests are permitted by the configured inference profile.

The training command validates only the historical training file. It does not load incoming batches for training. `House_ID` and `Batch` remain excluded from model features according to the existing config.

## 5. Tests and verification

Commands executed from the repository root:

```bash
python -m pytest -q
PYTHONPATH=src python -m mlops_pipeline validate-config
PYTHONPATH=src python -m mlops_pipeline check-contract
python -m compileall -q src tests
```

Results:

- `python -m pytest -q`: **224 passed**.
- `validate-config`: passed. It reported nine pending future model-gate/monitoring settings, as expected for later phases.
- `check-contract`: passed for training data (4000 rows) and incoming data (1500 rows).
- `compileall`: passed (no syntax/bytecode compilation errors).

The first full-suite run before the final fix had 221 passed and one failure: a nullable categorical column was inferred as `object` under the installed pandas version and incorrectly skipped its allowed-values check. `src/mlops_pipeline/data/contract.py` now accepts an object-backed text column when all non-null values are strings; the full suite subsequently passed.

New integration coverage: `tests/integration/test_quality_gate.py` mutates an in-memory copy of the training DataFrame to give `Area_sqft` an invalid negative value, replaces downstream split/evaluation/artifact-save functions with fail-on-call spies, and verifies that `DataValidationError` contains a failed numeric-range result and none of the downstream functions ran. A second test verifies validator-construction/configuration errors halt before splitting. Existing integration tests exercise the successful real-data training path.

## 6. Dataset integrity

No supplied CSV was edited by implementation or tests. The current repository dataset SHA-256 values are:

- `data/raw/capstone_house_prices_train.csv`: `8696565fd2a80775efee9574463c81877a23cbf51314a72e5919c81936036a3c`
- `data/raw/capstone_incoming_batches.csv`: `94c1c5aeac37d19d620b0698f7b65dcc792e2dc2196d4e68ced356921359318e`

The bad-data integration fixture is held in memory only.

## 7. Production-safety guarantees and limitations

- A failed report raises before split, fitting, evaluation, or artifact persistence.
- Unexpected errors constructing/running the validator propagate and stop execution before downstream training side effects; individual validator check exceptions are converted to failed check results rather than silent passes.
- Existing production model files are not replaced on validation failure because the save path is reachable only after the gate.
- The gate does not provide atomic writes for failures that occur later during training/saving; that is outside the data-validation gate's guarantee.
- The current data contract runs during loading and may reject structural problems before the richer validator report can be produced. This is fail-safe, but callers may see `ContractViolationError` for those earlier failures.
- Thresholds are plausibility/quality thresholds, not drift detection thresholds. Incoming distributions may legitimately differ; drift monitoring remains future work.

## 8. Files added or changed in this completion

- `src/mlops_pipeline/models/training.py` — invoke the quality gate before split/training/artifact writes.
- `src/mlops_pipeline/cli.py` — display structured validation failures and exit non-zero.
- `src/mlops_pipeline/data/contract.py` — handle nullable object-backed categorical text correctly.
- `tests/integration/test_quality_gate.py` — fail-closed integration tests.
- `PHASE_3_HANDOFF.md` — this handoff document.

The supplied Phase 3A archive already contained the validator, config rules, validator unit tests, real-data tests, and `docs/DATA_VALIDATION.md`; those were retained.

## 9. Phase 4 instructions (MLflow/Prefect deferred)

Do not add MLflow or Prefect as part of Phase 3. In Phase 4:

1. Keep `run_training`'s validation gate as the first post-load stage.
2. If introducing stage-wise orchestration, wrap existing functions rather than rewriting them: `load_training_data`, `DataValidator.validate`, `split_train_validation`, `evaluate_model`, `check_leakage`, and the current artifact/report save logic.
3. Add Prefect orchestration under `src/mlops_pipeline/pipeline/` only when Phase 4 scope is approved; validation failure must propagate as a failed task and block downstream tasks.
4. Add MLflow tracking only in the tracking phase, recording YAML parameters, metrics, config/data hashes, and model artifacts. A failed data gate must not create a successful training run or register/promote a model.
5. Keep future model-gate and monitoring settings in YAML and validate them through `schema.py` / config validation; do not hardcode thresholds.
6. Preserve the rule that incoming batches are not training inputs and `House_ID` / `Batch` are not model features.
