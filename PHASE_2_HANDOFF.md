# PHASE 2 HANDOFF - ML Foundation

Project 4 MLOps capstone (house price prediction). Scope of this phase: the ML
foundation only. Not implemented, by instruction: Prefect, MLflow, model
registry, FastAPI, Docker, CI/CD, monitoring, data/model gates.

Every number below was produced by `python -m mlops_pipeline train` (seed 42) and
is stored in `reports/model_evaluation.json`. Nothing is estimated or copied
from an earlier session.

## 0. Provenance note (read first)

When this phase started, the working directory already contained a partial
Phase 2 build on top of Phase 1 (code, tests, saved models, a report, and
`model.selected: baseline` with a justification already written into the YAML).
It was not produced in this session, so it was treated as unverified:

- all code was reviewed; the Phase 1 files were not rebuilt;
- all generated outputs (models, report) were deleted and regenerated;
- `model.selected` was reset to `null`, the comparison was run, and the
  selection was made only after reading the results (section 4);
- the metrics were recomputed independently with plain scikit-learn (section 6).

## 1. What was implemented

| # | Requirement | Where |
|---|---|---|
| 1 | Data loading (training file only; contract-checked; rejects a `Batch` column) | `data/loading.py` |
| 2 | Train/validation split (seeded, 80/20, split happens first) | `data/splitting.py` |
| 3 | Feature/target separation (configured columns only) | `data/splitting.py` |
| 4 | `House_ID` and `Batch` excluded from features | config `columns.excluded`; `remainder="drop"` in the preprocessor; explicit guard |
| 5 | Missing values: median imputation (Age_Years, Distance_to_Center_km, Neighborhood_Score) | `features/preprocessing.py` |
| 6 | Encoding: ordinal for Property_Condition (Poor<Fair<Good<Excellent), one-hot for City_Zone, unknown categories ignored | `features/preprocessing.py` |
| 7 | Reproducible preprocessing: one config-driven `ColumnTransformer` inside one sklearn `Pipeline` | `models/factory.py` |
| 8 | Baseline: linear regression (OLS) | config `model.baseline` |
| 9 | One candidate: gradient boosting (300 trees, lr 0.05, depth 3; fixed in advance, not tuned) | config `model.candidate` |
| 10 | Evaluation module: MAE, RMSE, R2; identical method for both models | `models/evaluation.py` |
| - | Machine-checkable leakage checks (8) | `models/leakage.py` |
| - | Training run, report, model saving; CLI `train` | `models/training.py`, `cli.py` |

Evaluation method (same for both models): fit on the training split only;
5-fold CV on the training split only (pipeline cloned and refitted inside each
fold); one scoring on the untouched validation split; training-split scores to
expose over-fitting. Primary metric (`mae`) was fixed in the config before results.

Data: 4000 rows -> 3200 train / 800 validation. Incoming batches (1500 rows) are
never read; a test spies on `pandas.read_csv` to prove it.

## 2. Actual results

Source: `reports/model_evaluation.json`. Units: MAE and RMSE in Rs lakhs.

| Model | Split | MAE | RMSE | R2 |
|---|---|---:|---:|---:|
| linear_regression_baseline | train (3200) | 5.6549 | 7.4719 | 0.8860 |
| | 5-fold CV mean (train split) | 5.6820 | 7.5058 | 0.8845 |
| | **validation (800)** | **5.7931** | **7.5044** | **0.8831** |
| gradient_boosting_candidate | train (3200) | 4.7481 | 6.3003 | 0.9190 |
| | 5-fold CV mean (train split) | 5.8569 | 7.7647 | 0.8762 |
| | **validation (800)** | **5.7777** | **7.5846** | **0.8806** |

Context: predicting the training mean gives validation MAE 17.41; the
validation target standard deviation is 21.96. Both models are far better than
that.

Extra comparison evidence:

| Evidence | Result |
|---|---|
| Validation MAE, candidate minus baseline | -0.0154 lakhs (0.27% better for candidate) |
| Paired bootstrap 95% CI of that difference (5000 resamples, 800 validation rows) | [-0.2115, +0.1956]; contains 0 |
| Validation RMSE / R2 | baseline better on both |
| CV MAE (mean +/- std over 5 folds) | baseline 5.682 +/- 0.056; candidate 5.857 +/- 0.139 |
| CV MAE, same folds, candidate minus baseline | +0.175 mean, std 0.114; candidate better in 1 of 5 folds |
| Over-fitting (train MAE vs CV MAE) | baseline 5.65 vs 5.68; candidate 4.75 vs 5.86 |

## 3. Selected model

`model.selected: baseline` (linear regression). The complete fitted
preprocessing + model pipeline is `artifacts/models/selected_model.joblib`.
Both fitted pipelines are also saved next to it in `artifacts/models/`.

## 4. Model selection reasoning

The rule fixed before the run was "primary metric = MAE". On the single
validation split the candidate has the lowest MAE, so by that rule alone it
would win. The baseline was selected anyway, deliberately, because:

1. The MAE edge is 0.0154 lakhs (0.27%) and its bootstrap CI straddles zero. On
   this validation set the two models are statistically indistinguishable.
2. The baseline is better on validation RMSE and R2.
3. The baseline is better on cross-validation MAE (5.68 vs 5.86), in 4 of 5
   paired folds, and with a smaller spread. CV uses 3200 rows over 5 folds and is
   a steadier basis than one 800-row split.
4. The candidate over-fits (train MAE 4.75 vs CV MAE 5.86).
5. With equal accuracy, the simpler and explainable model wins. The largest
   coefficients: City_Zone_Central +14.2, City_Zone_North -10.9, City_Zone_East
   -7.9, Has_Parking +6.2, Property_Condition (ordinal step) +6.1,
   Neighborhood_Score +4.9, Bedrooms +3.2 (lakhs, other features fixed).

The report records this honestly: `selection.matches_best_on_validation = false`.

Limits of this conclusion: the candidate's hyperparameters were a single
untuned default. This shows that this gradient boosting setting is no better
than a linear model on this data, not that no tuned tree model could be. The
validation split was looked at while choosing, so its numbers are a slightly
optimistic estimate of future performance; the CV numbers were not used for
the choice of hyperparameters, and no model was tuned. The selected model is
fitted on the 3200 training rows only, not refitted on all 4000.

## 5. Leakage check

All 8 automated checks pass (`reports/model_evaluation.json -> leakage_checks`;
`train` exits non-zero if any fails):

| Check | Result |
|---|---|
| House_ID, Batch, target absent from X_train / X_valid | pass |
| Fitted pipelines (inputs and output feature names, both models) contain none of them | pass (2 checks) |
| Train and validation disjoint by House_ID and together complete (3200 + 800 = 4000) | pass |
| Fitted median-imputation values equal the median of the training split (both models) | pass (2 checks); using train+validation would change some of the values, so the check can detect leakage |
| One-hot categories learned from the training split only (both models) | pass (2 checks) |

Further evidence from tests: CV refits the whole pipeline in every fold with the
held-out rows never seen; predictions are identical whether or not `House_ID`,
`Batch` or the target are passed to the saved pipeline; incoming file is never
read; leakage checks are tested to fail when an imputer is fitted on validation
rows, when the split overlaps, and when an identifier enters X.

Not covered by the checks (be aware): there is no time column, so a random
split cannot test temporal leakage; and the checks cannot prove a feature is not
derived from the price (the brief's data dictionary gives no sign of that).

## 6. Verification performed

- `python -m pytest`: **131 passed** (62 from Phase 1 + 69 new/updated), about 17 s.
- Training run twice: identical metrics (deterministic).
- Independent recompute with plain scikit-learn (own split call, own encoding):
  baseline matches exactly (5.7931 / 7.5044 / 0.8831); candidate within 0.0005
  (5.7775 / 7.5841 / 0.8806), the gap coming from a different column order that
  changes tree tie-breaks.
- Saved pipeline loaded in a fresh Python process: predicts from raw rows that
  contain `House_ID` and `Price_Lakhs` (same output as without them), and handles
  an unseen City_Zone and a NaN.
- Missingness check on the training split only: missing vs present mean price
  62.8/62.1 (Age_Years), 61.5/62.1 (Distance), 63.9/62.1 (Neighborhood_Score),
  Welch p = 0.72 / 0.73 / 0.38. No evidence it is informative, so no missing-
  indicator columns were added.

## 7. Files created / modified in Phase 2 (relative to Phase 1)

Created (source):
- `src/mlops_pipeline/data/loading.py`
- `src/mlops_pipeline/data/splitting.py`
- `src/mlops_pipeline/features/preprocessing.py`
- `src/mlops_pipeline/models/factory.py`
- `src/mlops_pipeline/models/evaluation.py`
- `src/mlops_pipeline/models/leakage.py`
- `src/mlops_pipeline/models/training.py`

Created (tests):
- `tests/unit/test_splitting.py`, `test_preprocessing.py`, `test_evaluation.py`,
  `test_leakage_checks.py`, `test_model_factory.py`, `test_config_model.py`
- `tests/integration/test_training_run.py`

Created (generated, git-ignored, regenerated by `train`):
- `artifacts/models/selected_model.joblib`
- `artifacts/models/linear_regression_baseline.joblib`
- `artifacts/models/gradient_boosting_candidate.joblib`
- `reports/model_evaluation.json`

Created (docs): `PHASE_2_HANDOFF.md`

Modified:
- `config/config.yaml` (new `paths` entries; `preprocessing` confirmed; `model`
  section filled: baseline, candidate, selected, split, evaluation)
- `src/mlops_pipeline/config/schema.py`, `config/validation.py` (model section)
- `src/mlops_pipeline/cli.py` (`train` command)
- `src/mlops_pipeline/data/__init__.py`, `features/__init__.py`, `models/__init__.py` (exports)
- `requirements.txt` (+ scikit-learn, joblib)
- `Makefile` (`train` target), `README.md`
- `tests/unit/test_config_loader.py` (pending list reflects Phase 2)

Unchanged: raw CSVs (byte-identical to the upload), `contract.py`, loader,
`pyproject.toml`, Phase 1 tests other than the one above.

Pending settings: 15 -> 11 (the four resolved: algorithm/hyperparameters
replaced by `model.baseline`/`model.candidate`, `test_size`, `primary_metric`).

## 8. Exact commands to reproduce

```bash
cd mlops-house-price-pipeline
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt        # + pytest
pip install -e . --no-deps
python -m mlops_pipeline validate-config
python -m mlops_pipeline check-contract
python -m mlops_pipeline train             # prints the table in section 2, rewrites models + report
python -m pytest                           # 131 passed
```
Without installing the package: prefix commands with `PYTHONPATH=src`
(`make train`, `make test` do this). Versions used: Python 3.12.3,
scikit-learn 1.8.0, pandas 3.0.2, numpy 2.4.4, pydantic 2.14.0, PyYAML 6.0.3,
joblib 1.5.3, pytest 9.1.1. Other versions are untested (the lower bounds in
requirements are not verified); small numeric differences are possible.
Load the model:
```python
import joblib
pipe = joblib.load("artifacts/models/selected_model.joblib")
pipe.predict(raw_rows)    # raw feature columns; extra columns are dropped by the pipeline
```

## 9. Known issues and flags

1. Selection deviates from the plain "lowest validation MAE" rule (section 4);
   change `model.selected` to `candidate` in the YAML and re-run `train` to
   override. Tests accept either value.
2. Model files are git-ignored and the repo is not under git (Phase 1 flag
   unchanged); there is no data versioning yet.
3. One random split and one candidate setting only, by design; no tuning.
4. Unscaled features and plain OLS: acceptable for accuracy, but coefficients
   are in raw units (Area_sqft is per square foot).
5. Phase 1 flag 7 still stands: incoming batches differ from train (e.g. mean
   Area_sqft 1150 vs 1152 / 1064 / 936). This model has not been evaluated on
   them; doing so belongs to monitoring and would be a held-out evaluation, not
   training data.
6. `random_seed: 42` and the 80/20 split are choices, not brief requirements.
7. Only Python 3.12 / scikit-learn 1.8 was tested.

## 10. Instructions for Phase 3

The brief's next unbuilt tasks are Task 3 (orchestrated, stage-wise pipeline),
Task 4 (gates), Task 5 (tracking, registry, lineage). The Phase 3 scope is the
project owner's call; the recommendation below is Task 3 plus experiment
tracking, with gates and the registry after it. Confirm before starting.

1. Start by running section 8 and confirming 131 tests pass and the metrics in
   section 2 reproduce.
2. Do not re-implement Phase 2 logic. Wrap it: expose `load_training_data`,
   `split_train_validation`, `build_pipeline`, `evaluate_model`,
   `check_leakage` and the saving code in `models/training.py` as separate
   stage functions with defined inputs/outputs (ingest, validate data, build
   features, train, evaluate), each re-runnable alone. Prefect goes in
   `src/mlops_pipeline/pipeline/`; add `prefect` to `requirements.txt` only then.
3. Add MLflow tracking for every run: parameters from the YAML, MLflow metrics =
   the MAE/RMSE/R2 values of this report, config file and data hash
   (`training_file_sha256` already in the report) as run records, the fitted
   pipeline as the model artifact. Add `mlflow` to `requirements.txt` then.
4. Any new setting goes into `config/config.yaml` first, then `schema.py`
   (and `validation.py`), then tests. Keep `test_no_hardcoding.py` green.
5. Keep the rules: only the training file trains the model; incoming batches stay
   out of training; House_ID and Batch never enter the model. Keep the
   leakage checks as part of the evaluate stage and fail the run if any fails.
6. Gates (Task 4) need the 11 remaining pending thresholds
   (`validation.missing_values.*`, `model_gate.*`, `monitoring.*`). Decide them
   with a recorded justification; the model gate can use the metrics in
   section 2 as the "current model" reference. Do not invent values.
7. Decide, and record in the YAML, whether the production model is refitted on
   all 4000 rows after selection (this phase did not).
8. Out of scope until later phases: registry transitions, FastAPI, Docker,
   CI/CD, Prometheus/Grafana/Evidently.
9. End the phase with `PHASE_3_HANDOFF.md` in the same format.
