# Project 4 — Phase 6 Handoff: Model Quality Gate

## Status

Implemented and tested the Model Quality Gate in the existing Prefect-oriented pipeline. The ML algorithms, feature engineering, split, preprocessing, model hyperparameters, and metric calculations were not changed.

- Full test suite: **235 passed** (`PYTHONPATH=src python -m pytest -q`).
- Configuration validation: passed (`PYTHONPATH=src python -m mlops_pipeline validate-config`).
- Successful pipeline-path demonstration: 4,000 rows; candidate passed both gate checks; production model SHA-256 remained unchanged.
- Controlled poor-candidate demonstration: gate returned FAIL, recorded candidate and production metrics/reasons, and preserved the production artifact hash.
- Prefect is not installed in this execution environment, so orchestration tests exercised the project's local task-compatible fallback. The flow code uses Prefect tasks when the dependency is installed. This phase did not verify the Prefect UI.
- MLflow is not installed in this environment. The isolated end-to-end verification disabled MLflow in a temporary copy; existing Phase 5 tracking behavior was not redesigned. Install dependencies to verify combined live Prefect/MLflow behavior.

## Gate logic

The gate is a distinct `7-model-quality-gate` task after evaluation:

```text
Ingestion
   ↓
Validation
   ↓
Data Gate ── fail ──> stop; no training and no model gate
   ↓ pass
Feature Engineering → Training → Evaluation
                                  ↓
                          Candidate metrics
                                  ↓
                      1. Absolute quality check
                                  ↓
                      2. Production comparison
                                  ↓
                           PASS or FAIL
```

Both checks are reported independently. The primary metric is read from `model.evaluation.primary_metric`; the gate does not hardcode `mae` in its decision logic. The metric's configured direction is respected (MAE/RMSE lower is better; R² higher is better).

### Check 1 — Absolute quality threshold

Config key: `model_gate.quality_threshold`

Configured value: `5.793117272533898` for MAE.

Rationale: the threshold is the **actual validation MAE of the existing production model** recorded in the repository's Phase 2 evaluation report, using the fixed seed-42 train/validation split. It is an evidence-derived baseline, not an arbitrary percentage or a guessed value. The value is in the target's native unit (Rs lakhs). A candidate passes this check if its primary metric is at least as good as the threshold (`MAE <= threshold`).

The original reported production baseline metrics are:

| Metric | Current production baseline |
|---|---:|
| MAE | 5.793117272533898 |
| RMSE | 7.504375060610438 |
| R² | 0.8830883392705375 |

### Check 2 — Comparison with current production

Config keys:

- `model_gate.production_comparison.enabled`
- `model_gate.production_comparison.min_improvement`
- `model_gate.production_metadata_path`

`min_improvement: 0.0` means the candidate must be **strictly better** than the current production model on the primary metric; equal values do not count as improvement. The two checks are independent: a candidate can meet the absolute threshold and still fail because it is worse than production.

The production comparison uses the metric recorded in the stable production metadata file. Before comparison, the pipeline verifies the SHA-256 of `artifacts/models/selected_model.joblib` against the metadata. If the artifact and metadata disagree, the pipeline fails closed instead of trusting stale production metrics.

### First-model case

If neither the configured production artifact nor production metadata exists, the gate records production comparison as `SKIPPED` with an explicit first-model explanation. The candidate can pass only if the absolute quality check passes. If a production artifact exists but metadata is missing, or metadata exists but the artifact is missing/hash-mismatched, the pipeline raises an error; it does not silently treat that state as a first-model case.

## Decision/report format

`reports/model_gate.json` contains:

- `status`: `PASS` or `FAIL`, plus boolean `passed`
- `primary_metric`
- all `candidate_metrics`
- `absolute_quality_check`: status, observed metric, threshold, direction, and reason
- `production_comparison`: status (`PASS`, `FAIL`, or `SKIPPED`), candidate value, production value, all production metrics, minimum improvement, model identity, and reason
- `failure_reasons`
- `candidate_disposition`: `eligible_for_future_promotion` or `rejected`
- `promotion_performed: false`
- `production_model_replaced: false`
- `production_model_preserved: true`

The evaluation report (`reports/model_evaluation.json`) is also updated with the gate result, and the updated report is attached to the Phase 5 MLflow runs when MLflow is enabled.

## Production safety

- The gate itself is a pure decision function and never saves/promotes models.
- The Prefect evaluation task can write baseline/candidate artifacts but no longer writes `selected_model.joblib`.
- The legacy `run_training()` API also no longer overwrites the selected production artifact; it may save candidate artifacts and reports only.
- Passing the gate does **not** promote the candidate. Registry promotion/deployment are explicitly out of scope until a later phase.
- Failing the gate returns `status: model_gate_failed` from the flow, writes the inspectable rejection report, and makes the CLI return a non-zero exit code.
- Data Gate still executes before production metadata lookup, feature engineering, and training. Invalid data cannot create a training run or invoke the Model Gate.

## Evidence / reproducible checks

### Commands

```bash
pip install -r requirements.txt
pip install -e . --no-deps
PYTHONPATH=src python -m mlops_pipeline validate-config
PYTHONPATH=src python -m pytest -q
python -m mlops_pipeline orchestrate
```

When dependencies are installed, the last command uses Prefect tasks. In the current offline environment the successful full-pipeline check was run on an isolated temporary project copy with MLflow disabled; the real repository data and production artifact were not modified.

### Successful run evidence (observed)

- 4,000 training rows processed.
- Candidate model: `gradient_boosting_candidate`.
- Candidate MAE: `5.777729174694131`.
- Absolute threshold: `5.793117272533898` → PASS.
- Production MAE: `5.793117272533898`.
- Candidate improvement: `0.015388097839767` Rs lakhs → production comparison PASS.
- Flow status: `completed`.
- Before/after SHA-256 of the production model artifact matched.

Note: the candidate's RMSE and R² are worse than production in the existing evaluation. The gate uses the project's predeclared primary metric (MAE) as requested; it does not silently switch metrics after observing results.

### Failed run evidence (controlled poor-model scenario)

Integration test: `test_poor_candidate_is_rejected_and_real_production_artifact_is_preserved`.

The test runs the existing pipeline in an isolated temporary project with a copy of the real production artifact and its recorded metadata. After real model fitting, it injects a deterministic poor candidate MAE of `50.0` solely as the controlled gate input. This does not alter the production model or supplied datasets.

Expected/observed:
- Absolute quality check: FAIL.
- Production comparison: FAIL.
- Candidate MAE (`50.0`) and production metrics (MAE `5.793117272533898`) are inspectable.
- Failure reasons are present.
- Flow status is `model_gate_failed`.
- Production artifact SHA-256 is identical before and after the run.

### Unit test scenarios

`tests/unit/test_model_quality_gate.py` covers:

1. Candidate passes both checks.
2. Candidate fails the absolute threshold.
3. Candidate passes absolute quality but is worse than production.
4. No production model exists (first-model behavior).
5. Controlled poor-model rejection.

Integration tests additionally cover the full success path, controlled poor-candidate rejection, and Data Gate stopping before training.

### Screenshot/evidence instructions

1. Run the pipeline with dependencies installed and capture the Prefect flow/task view showing the `7-model-quality-gate` task.
2. Capture `reports/model_gate.json` for a PASS run.
3. Run the controlled poor-candidate integration test and capture its pytest result plus the generated `model_gate.json` from the test's temporary project if preserving test artifacts locally.
4. Show the production artifact SHA-256 before and after the controlled failure to prove it was preserved.
5. For first-model behavior, run in a temporary project with neither the selected production model nor its metadata; verify `production_comparison.status == "SKIPPED"` and the explicit first-model reason.
6. Do not use or modify the supplied datasets to create the poor-model demonstration.

## Files added/changed

- `config/config.yaml` — evidence-derived absolute threshold, strict improvement setting, and production metadata path.
- `src/mlops_pipeline/config/schema.py` — non-null, validated model-gate thresholds and metadata path.
- `src/mlops_pipeline/models/quality_gate.py` — pure two-check decision logic, production artifact hash verification, and gate report persistence.
- `src/mlops_pipeline/pipeline/prefect_flow.py` — Model Gate task after evaluation, gate decision/report integration, and no production replacement.
- `src/mlops_pipeline/models/training.py` — legacy training API no longer writes the selected production artifact.
- `src/mlops_pipeline/cli.py` — non-zero CLI exit when the Model Gate rejects a candidate.
- `artifacts/models/production_model_metadata.json` — production model metrics and verified artifact hash derived from the repository's existing report and selected model.
- `tests/unit/test_model_quality_gate.py` — four required cases plus controlled poor-model case.
- `tests/unit/test_config_loader.py` — gate threshold validation tests.
- `tests/integration/test_prefect_flow.py` — gate behavior and production-preservation assertions.
- `tests/integration/test_training_run.py` — confirms legacy training does not replace production.
- `PHASE_6_HANDOFF.md` — this handoff.

## Known issues and limitations

1. Prefect and MLflow are not installed in the current offline environment, so live Prefect UI and live MLflow artifact verification remain pending. The flow/task path and tracking integration are tested without those external services.
2. The production metadata is initialized from the existing selected model and its recorded evaluation metrics. If the production artifact is manually changed, its SHA-256 mismatch will halt the pipeline until metadata is reconciled through an authorized future release process.
3. The absolute MAE threshold is tied to the fixed seed-42 validation split and current dataset. If the evaluation dataset or methodology is intentionally changed, reassess and document the threshold rather than silently tuning it.
4. Candidate model artifacts and evaluation reports are still produced before the Model Gate decision; these are candidate/evaluation artifacts, not production artifacts. The selected production artifact is not replaced.
5. No Model Registry, promotion, deployment, serving, or rollback workflow is implemented.

## Phase 7 instructions

1. Inspect this handoff and existing Model Gate, production metadata, MLflow tracking, and Prefect flow.
2. If implementing Model Registry or promotion, make it a separate downstream task that runs **only** after `model_gate.passed == true`.
3. Never promote or replace production on a Data Gate or Model Gate failure. Preserve the current production artifact and record the failure reason.
4. Use configured model/run identifiers and artifact hashes for lineage; do not create MLflow runs manually after training.
5. Add tests proving both gates block registry/promotion and that promotion records the exact MLflow run ID, model version, and artifact lineage.
6. Keep rollback/previous-version preservation explicit and testable. Do not implement serving unless Phase 7's approved scope requests it.
