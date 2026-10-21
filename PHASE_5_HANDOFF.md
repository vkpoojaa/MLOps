# Project 4 — Phase 5 Handoff: MLflow Experiment Tracking

## Status and verification boundary

MLflow integration is implemented in the pipeline. The default configuration enables tracking and creates one MLflow run per configured model role **before that model is fitted**. Training, parameters, metrics, and artifacts are logged by pipeline code.

Automated tests: **228 passed in 23.21 seconds** (`PYTHONPATH=src python -m pytest -q`).

The real MLflow UI could not be launched or verified in this environment because MLflow is not installed and package installation failed due unavailable DNS/network access. Therefore, no real MLflow server/UI screenshot or genuine backend run ID is claimed. Tracking behavior is covered with a deterministic fake MLflow client; use the commands below in an environment where dependencies can be installed to verify the live UI.

## Architecture

```text
Prefect flow
  |
  +-- Ingestion
  +-- Validation
  +-- Data Gate -------------------- failure --> flow stops; no training run created
  |
  +-- Feature Engineering
  |
  +-- Training task
  |     |
  |     +-- create MLflow run: baseline (BEFORE fit)
  |     |     +-- log parameters, lineage tags, config, validation report
  |     |     +-- fit existing sklearn pipeline
  |     |     +-- log MAE / RMSE / R² and serialized sklearn pipeline
  |     |
  |     +-- create MLflow run: candidate (BEFORE fit)
  |           +-- same tracking contract
  |
  +-- Evaluation
        +-- write existing comparison/evaluation JSON
        +-- attach evaluation report to each existing run
        +-- include run IDs and lineage in evaluation JSON and flow summary
```

No model algorithm, feature set, split strategy, preprocessing, cross-validation, or selection behavior was redesigned.

## MLflow experiment structure

Values are controlled in `config/config.yaml`:

```yaml
mlflow:
  enabled: true
  tracking_uri: sqlite:///mlflow.db
  experiment_name: house-price-regression
  artifact_subdirectory: phase5
```

- Experiment: `house-price-regression`
- Backend: local SQLite database (`mlflow.db`)
- Runs: one per model role, named `<role>-<model-name>` (normally baseline and candidate)
- The run is opened inside `fit_and_track_model()` before its `fit_callable()` executes. A fitting/logging exception leaves the run failed instead of producing a successful training record.
- The Data Gate executes before the training task. If validation fails, no model training run is created by the pipeline.
- `artifact_subdirectory` is retained as configuration for phase-level organization; artifacts are grouped by MLflow artifact paths (`phase5/model`, `phase5/configuration`, `phase5/validation`, `phase5/evaluation`).

## Tracked parameters

Each model run logs:
- Model type/algorithm, model name, and role
- Model hyperparameters (flattened into MLflow parameters)
- Preprocessing settings from the configuration
- Split test size, random seed, primary metric, and CV fold count
- Dataset identifier and SHA-256
- Configuration SHA-256 and Git commit/code version when available

Structured values are JSON-encoded before logging. Values are truncated to MLflow's 500-character parameter limit; lineage hashes are also recorded as tags.

## Metrics

The run records validation metrics as `MAE`, `RMSE`, and `R2` (R²). The current project config defines MAE as primary and RMSE/R² as secondary metrics. The underlying metric calculations remain the existing scikit-learn implementation.

## Artifacts

- `model/`: serialized fitted scikit-learn pipeline, including preprocessing
- `configuration/`: exact `config.yaml` used for the run
- `validation/`: structured Phase 3 validation report
- `evaluation/`: final evaluation/comparison JSON attached during the evaluation stage

The evaluation JSON also records each model's MLflow run ID, dataset/config hashes, and code version when available.

## Lineage

- Dataset identifier: actual CSV filename consumed by ingestion, including `--data-path` overrides
- Dataset version: SHA-256 computed from the exact input file bytes
- Configuration: path and SHA-256 of the exact YAML file
- Code version: `GIT_COMMIT` environment variable if provided, otherwise `git rev-parse HEAD`; if no Git metadata is available, the report explicitly records `unavailable-no-git-commit`
- MLflow run ID: returned by the run created before fitting, stored in the model result, flow summary, and evaluation JSON

No dataset or configuration file is modified to compute hashes.

## Live MLflow UI verification status

**Not verified in this execution environment.** Direct import checks confirmed both MLflow and Prefect are absent. The pipeline command was executed and exited with code 1 before fitting because MLflow tracking is enabled but the MLflow package is unavailable. Therefore, no real MLflow run ID, MLflow backend database, or UI screenshot was produced here. The fake-client tests validate the tracking contract but are not substitutes for a live UI run.

The pipeline must be rerun after installing dependencies in a network-enabled environment; use the commands below and capture the UI evidence listed later in this handoff.

## Commands

From the repository root, install dependencies:

```bash
pip install -r requirements.txt
pip install -e . --no-deps
```

Run the pipeline (creates MLflow runs from inside the training stage):

```bash
python -m mlops_pipeline orchestrate
```

Run the full test suite:

```bash
python -m pytest -q
```

Start the local MLflow UI against the configured SQLite store:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Open `http://127.0.0.1:5000`, select the `house-price-regression` experiment, and inspect the baseline and candidate runs. Depending on the installed MLflow version, the command may also be run as:

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

If the configured tracking URI is resolved to an absolute SQLite path by the application, use that same absolute database path in the UI command when launching from a different working directory.

## Screenshot / evidence instructions

Capture these after running in an environment with MLflow installed:

1. Experiment page showing `house-price-regression` and both model runs.
2. A run's Parameters view showing algorithm, hyperparameters, preprocessing, dataset/config hashes, and code version.
3. Metrics view showing MAE, RMSE, and R2.
4. Artifacts view showing `phase5/model/`, `phase5/configuration/`, `phase5/validation/`, and `phase5/evaluation/`.
5. Run details showing its run ID and tags.

For failed-data evidence, run the pipeline with a temporary CSV containing an invalid categorical value:

```bash
python -m mlops_pipeline orchestrate --data-path /path/to/temporary_invalid_training_data.csv
```

Expected: non-zero exit, Data Gate failure, no training task/run created by that invocation. Keep the supplied source datasets untouched. A failed-data attempt should not be shown as a successful model-training run.

## Tests and observed results

Command:

```bash
python -m pytest -q
```

Observed result: **228 passed**.

Coverage includes:
- Existing Phase 1–4 tests
- Tracking parameters, metrics, artifacts, hashes/tags, and run ID using a fake MLflow client
- Assertion that the MLflow run is opened before the fit callback
- Assertion that tracking initialization failure prevents fitting
- Existing Prefect flow tests for valid data and Data Gate rejection before training
- Configuration validation with the new MLflow section

Configuration check:

```bash
PYTHONPATH=src python -m mlops_pipeline validate-config
```

Observed: configuration valid.

Live pipeline/UI limitation:

```bash
PYTHONPATH=src python -m mlops_pipeline orchestrate
```

Observed in this environment: exits with `RuntimeError: MLflow tracking is enabled in config.yaml but MLflow is not installed`. Installing MLflow was attempted but failed because the package index could not be reached (DNS/network error). This fail-closed behavior is intentional: the pipeline does not silently train without tracking when tracking is enabled. A real MLflow UI run remains unverified until dependencies can be installed.

## Files changed

- `config/config.yaml` — MLflow tracking settings
- `requirements.txt` — added `mlflow>=2.12,<4`; retained Prefect dependency
- `src/mlops_pipeline/config/schema.py` — typed MLflow configuration schema
- `src/mlops_pipeline/monitoring/mlflow_tracking.py` — MLflow run lifecycle, parameters, metrics, artifacts, hashes, and code lineage
- `src/mlops_pipeline/pipeline/prefect_flow.py` — create runs before fitting, pass validation and dataset lineage, log evaluation artifacts, expose run IDs
- `tests/integration/test_prefect_flow.py` — isolate existing flow/gate tests from unavailable MLflow dependency
- `tests/unit/test_mlflow_tracking.py` — tracking contract and fail-closed tests
- `PHASE_5_HANDOFF.md` — this document

## Known issues / limitations

1. Live MLflow UI verification and real backend run IDs are pending because the package could not be installed in this environment.
2. If tracking is enabled and MLflow is missing/unreachable, the pipeline fails before fitting rather than silently proceeding untracked.
3. If the repository has no Git metadata and `GIT_COMMIT` is not supplied, code version is explicitly marked unavailable.
4. MLflow Registry promotion, Model Gate, deployment, and serving are not implemented in this phase.

## Phase 6 instructions

1. Inspect this handoff and the existing config, tracking helper, Prefect flow, and evaluation report.
2. Implement the configured Model Gate as a distinct step after evaluation, using only thresholds/settings defined in `config.yaml`.
3. If Phase 6 introduces Model Registry or promotion, make it strictly downstream of a passing Model Gate; do not implement promotion in Phase 5.
4. Ensure a failed Data Gate creates no model-training run and a failed Model Gate cannot trigger registration/promotion.
5. Add success/failure tests that prove downstream calls are not made after either gate fails.
6. Preserve current model methodology and the MLflow lineage already recorded.
