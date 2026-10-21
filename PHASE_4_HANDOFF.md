# Project 4 — Phase 4 Handoff: Prefect Orchestration

## Status

The orchestration implementation is complete and the end-to-end success and failure paths were executed in an isolated copy of the repository. The current execution environment did **not** have Prefect installed, and package installation was blocked by unavailable network/DNS. Therefore, these runs used the module's explicitly logged local fallback decorators; they verify stage logic and the gate behavior, but are **not** evidence of a Prefect-server/UI run. With `prefect` installed, the same flow and tasks use Prefect's real decorators.

## Architecture

`src/mlops_pipeline/pipeline/prefect_flow.py` wraps the existing config, validator, split, model factory, evaluation, leakage checks, and report/artifact-writing code. It does not redesign model algorithms or preprocessing.

### Flow diagram

```text
load config.yaml
      |
      v
[1 Ingestion] -- read configured training CSV
      |
      v
[2 Validation] -- DataValidator / train profile
      |
      v
[3 Data Gate] -- raise DataValidationError if any check fails
      |
      v
[4 Feature Engineering] -- configured feature/target separation + seeded split
      |
      v
[5 Training] -- fit baseline + candidate using existing sklearn pipelines
      |
      v
[6 Evaluation] -- leakage checks, comparison, report + model artifacts
```

The task dependencies are sequential. A failed validation report raises in the Data Gate, which fails the flow before feature engineering, training, or evaluation can be called. Preprocessing remains inside each existing sklearn `Pipeline` and is fitted only on training folds/split, preserving the prior leakage protections.

## Prefect tasks

| Stage | Task | Inputs | Outputs / side effects |
|---|---|---|---|
| 1 | `1-ingestion` | Validated settings; optional input CSV override | Raw pandas DataFrame; logs source path and dimensions |
| 2 | `2-validation` | DataFrame, settings | Structured `ValidationReport`; logs counts and individual failures |
| 3 | `3-data-gate` | DataFrame, validation report | Same DataFrame only when valid; otherwise raises `DataValidationError` |
| 4 | `4-feature-engineering` | Gated DataFrame, settings | Existing `DataSplit` with train/validation matrices and targets |
| 5 | `5-training` | DataSplit, settings | Fitted baseline and candidate `ModelResult` objects |
| 6 | `6-evaluation` | Frame, split, model results, settings | Leakage checks, model comparison, JSON evaluation report, model files, run summary |

Configuration is loaded through the existing `load_config` and `config/config.yaml` path resolution. `--config` can select an alternate config. `--data-path` is an explicit input override for repeatable testing/batch input; it does not modify the configured CSV.

## Commands

Install project dependencies in the target environment:

```bash
pip install -r requirements-dev.txt
pip install -e . --no-deps
```

Run the orchestrated flow:

```bash
python -m mlops_pipeline orchestrate
```

Use a specific configuration or test input:

```bash
python -m mlops_pipeline orchestrate --config config/config.yaml
python -m mlops_pipeline orchestrate --config config/config.yaml --data-path /path/to/input.csv
```

Run tests:

```bash
python -m pytest -q
```

### Prefect UI visibility

Once dependencies are installed, start a local Prefect server in one terminal:

```bash
prefect server start
```

In another terminal, configure the local API and run the flow:

```bash
prefect config set PREFECT_API_URL=http://127.0.0.1:4200/api
python -m mlops_pipeline orchestrate
```

Open `http://127.0.0.1:4200` to inspect flow runs and task states. The flow/task decorators are real Prefect decorators whenever Prefect is installed. If it is absent, the CLI emits a warning and uses a local fallback solely so validation and pipeline behavior can still be tested; this fallback does not create Prefect UI records.

## Execution evidence

### SUCCESS — valid data

Command run in an isolated copy of the project:

```bash
PYTHONPATH=src python -m mlops_pipeline orchestrate --config /tmp/project4_phase4_demo/config/config.yaml
```

Observed exit code: `0`.

Observed summary:

- Status: `completed`
- Rows: `4000`
- Split: `3200` train / `800` validation
- Primary metric: MAE
- Baseline validation: MAE `5.7931172725`, RMSE `7.5043750606`, R² `0.8830883393`
- Candidate validation: MAE `5.7777291747`, RMSE `7.5845565477`, R² `0.8805766763`
- Leakage checks passed: `8/8`
- Outputs: `artifacts/models/linear_regression_baseline.joblib`, `artifacts/models/gradient_boosting_candidate.joblib`, `artifacts/models/selected_model.joblib`, `reports/model_evaluation.json`

The configured selected model remains `baseline`; the evaluation report separately records that `candidate` had the lowest validation MAE. This preserves the Phase 2 selection decision rather than silently changing it.

### FAILURE — invalid category

A test-only copy of the training CSV was modified in `/tmp` to contain `INVALID_ZONE_FOR_PHASE4_DEMO` in `City_Zone`. The supplied source CSV was not edited.

Command:

```bash
PYTHONPATH=src python -m mlops_pipeline orchestrate --config /tmp/project4_phase4_demo/config/config.yaml --data-path /tmp/project4_phase4_demo/invalid_training_data.csv
```

Observed exit code: `1`.

Observed failure details:

- Validation profile: `train`
- Check: `allowed_values`
- Column: `City_Zone`
- Invalid values: `['INVALID_ZONE_FOR_PHASE4_DEMO']`
- Data Gate logged `DATA GATE BLOCKED: downstream feature engineering and training will not run`.
- The flow raised `DataValidationError`. This CLI failure run followed the successful run in the same isolated demo directory, so prior outputs already existed; the failure was blocked at the gate.
- Automated integration test runs the failure path in a fresh isolated project, replaces the training task with a fail-if-called spy, confirms training is not invoked, and verifies pre-existing model/report files remain byte-for-byte unchanged.

**Execution limitation:** because Prefect could not be installed in this network-restricted environment, successful/failed command evidence is for the local fallback path, not Prefect UI history. Run the commands in the Prefect-enabled environment to verify UI visibility.

## Test results

Command:

```bash
python -m pytest -q
```

Result: **226 passed** in `22.31s` (final verification run).

The new `tests/integration/test_prefect_flow.py` includes:

1. Valid-data end-to-end success in an isolated temporary project.
2. Invalid-category gate failure, inspectable validation report, training task not called, and no output artifacts created.

All tests were run after fixing a hardcoding-policy test that initially flagged an output dictionary key. No test failures remain.

## Files added/changed

- `requirements.txt` — added `prefect>=3.0,<4`.
- `src/mlops_pipeline/pipeline/prefect_flow.py` — six orchestration tasks, flow definition, logs, and local fallback for environments where Prefect is unavailable.
- `src/mlops_pipeline/pipeline/__init__.py` — updated package description.
- `src/mlops_pipeline/cli.py` — added `orchestrate` command with config/input overrides and a warning if Prefect is missing.
- `tests/integration/test_prefect_flow.py` — success/failure orchestration tests.
- `PHASE_4_HANDOFF.md` — this handoff.

No supplied dataset was edited by the demonstration; all demo modifications and generated outputs were isolated under `/tmp/project4_phase4_demo`.

## Known issues and limitations

1. Prefect could not be installed in this execution environment because package index access failed with DNS/network errors. The actual Prefect decorators/UI visibility must be verified after installing `requirements.txt` in an internet-enabled environment.
2. No Prefect deployment, work pool, schedule, Model Gate, MLflow Registry, or deployment was created; these are outside Phase 4 scope.
3. The optional `--data-path` is intended for explicit alternate inputs and tests. Normal operation uses the configured training CSV.
4. Model fitting and cross-validation remain the existing sklearn implementation. Preprocessing is performed within sklearn pipelines to retain train-fold isolation.

## Phase 5 instructions

1. Inspect this handoff and the existing flow/tasks before making changes.
2. Implement the configured Model Gate as a distinct stage after evaluation, before any future registration/promotion.
3. Read `model_gate` thresholds and comparison settings from `config/config.yaml`; fail safely if required settings remain pending.
4. Test both passing and failing model-gate outcomes without changing the existing modelling algorithms.
5. Do **not** implement MLflow Registry or deployment unless Phase 5's official brief explicitly includes them.
6. Preserve the Prefect flow's explicit task dependencies and ensure failed gates block all downstream side effects.
