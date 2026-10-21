# PHASE 1 HANDOFF - Project Foundation

Project 4 MLOps capstone (house price prediction). Scope of this phase: project
foundation only. No model, MLflow, Prefect, registry, API, Docker, CI/CD,
Prometheus/Grafana, drift monitoring or deployment exists yet, and no model
results have been produced.

## 1. What was implemented

- Repository structure (src layout, tests, docs, placeholders for later phases).
- `config/config.yaml` as the single source of truth: paths, target, feature and
  excluded columns, data contract, preprocessing, model, data-gate, model-gate
  and monitoring settings.
- Typed configuration schema (pydantic v2, unknown keys rejected).
- Configuration loader (`load_config`) with YAML duplicate-key detection,
  `MLOPS_CONFIG` override and optional data-file existence check.
- Configuration validation: type/range checks plus cross-section consistency
  (column roles vs. contract, encoders vs. features, relative in-repo paths).
- Data contract defined in YAML and interpreted by `DataContract` (structural
  conformance report only; this is **not** the pipeline data gate).
- CLI: `validate-config`, `check-contract`.
- README, `.gitignore`, `requirements.txt`, `requirements-dev.txt`,
  `pyproject.toml`, `Makefile`, test suite (62 tests).

Datasets were inspected again at the start of this session (no earlier
inspection was available in my context); all facts below were re-verified from
the files.

## 2. Final directory structure

```
mlops-house-price-pipeline/
├── config/config.yaml
├── data/
│   ├── raw/                      capstone_house_prices_train.csv, capstone_incoming_batches.csv
│   ├── interim/                  (git-ignored contents)
│   └── processed/                (git-ignored contents)
├── docs/                         official Project 4 brief (.docx)
├── src/mlops_pipeline/
│   ├── __init__.py, __main__.py, cli.py
│   ├── config/   __init__.py, schema.py, loader.py, validation.py
│   ├── data/     __init__.py, contract.py
│   └── pipeline/ features/ models/ serving/ monitoring/   (empty placeholders)
├── tests/
│   ├── conftest.py
│   ├── unit/         test_config_loader.py, test_config_validation.py,
│   │                 test_data_contract.py, test_no_hardcoding.py
│   └── integration/  test_real_data.py
├── artifacts/ logs/ reports/ notebooks/ scripts/ docker/ .github/workflows/   (.gitkeep only)
├── .gitignore, Makefile, pyproject.toml, README.md
├── requirements.txt, requirements-dev.txt
└── PHASE_1_HANDOFF.md
```

## 3. Configuration structure (`config/config.yaml`)

| Section | Purpose | State |
|---|---|---|
| `config_version`, `project`, `reproducibility` | metadata, `random_seed: 42` | set (seed is a baseline default) |
| `paths` | train/incoming CSVs, interim/processed/artifacts/logs/reports dirs, all repo-relative | set |
| `columns` | `target: Price_Lakhs`; `features.numeric` (7), `features.categorical` (2); `excluded` = House_ID (role identifier), Batch (role batch) | set |
| `data_contract` | per-column dtype, nullable, unique, allowed_values, min/max, `available_in` | set (see section 4) |
| `preprocessing` | median imputation, scaling none, ordinal encoding for Property_Condition, one-hot for City_Zone, `handle_unknown: ignore` | **baseline defaults, to confirm** |
| `model` | `problem_type: regression`; algorithm, hyperparameters, `split.test_size`, `primary_metric` | pending (null) |
| `validation` (data gate) | check toggles on; missing-value limits; `on_failure: halt` | limits pending (null) |
| `model_gate` | `quality_threshold`, `production_comparison.enabled: true`, `min_improvement`, `on_failure: halt` | thresholds pending (null) |
| `monitoring` | `reference_dataset: train`; drift method/thresholds, performance-drift, latency/error limits, `on_alert` | pending (null) |

Pending settings (15, listed by `validate-config`): `model.algorithm`,
`model.hyperparameters`, `model.split.test_size`, `model.evaluation.primary_metric`,
`validation.missing_values.max_overall_fraction`,
`validation.missing_values.max_fraction_per_column`, `model_gate.quality_threshold`,
`model_gate.production_comparison.min_improvement`, `monitoring.data_drift.method`,
`monitoring.data_drift.per_feature_threshold`,
`monitoring.data_drift.drifted_feature_share_threshold`,
`monitoring.performance_drift.max_degradation`,
`monitoring.operational.latency_p95_ms_max`, `monitoring.operational.error_rate_max`,
`monitoring.on_alert`.

Metric used by the model gate is defined once, in `model.evaluation.primary_metric`.

Python access: `load_config()` returns `Settings` (`.config` typed `AppConfig`,
`.train_path`, `.incoming_path`, `.resolve()`, `.pending`). Derived values:
`cfg.target`, `cfg.feature_columns`, `cfg.numeric_features`,
`cfg.categorical_features`, `cfg.excluded_columns`, `cfg.identifier_column`,
`cfg.batch_column`, `cfg.batch_values`.

## 4. Data contract

Verified against the real files: both conform.

| Column | Type | Nullable | Constraint | In train | In incoming | Role |
|---|---|---|---|:-:|:-:|---|
| House_ID | string | no | unique | yes | yes | excluded (identifier) |
| Batch | integer | no | values {1,2,3} | no | yes | excluded (batch metadata) |
| Area_sqft | integer | no | | yes | yes | numeric feature |
| Bedrooms | integer | no | | yes | yes | numeric feature |
| Bathrooms | integer | no | | yes | yes | numeric feature |
| Age_Years | integer-valued | **yes** (160 / 60 missing) | | yes | yes | numeric feature |
| Distance_to_Center_km | float | **yes** (160 / 60) | | yes | yes | numeric feature |
| Neighborhood_Score | float | **yes** (160 / 60) | 1 to 10 | yes | yes | numeric feature |
| Has_Parking | integer | no | values {0,1} | yes | yes | numeric (binary) feature |
| Property_Condition | categorical | no | Poor, Fair, Good, Excellent | yes | yes | categorical feature |
| City_Zone | categorical | no | North, South, East, West, Central | yes | yes | categorical feature |
| Price_Lakhs | float | no | | yes | yes | **target** |

Only constraints documented in the project brief are encoded. Min/max exist
only for Neighborhood_Score. Unexpected columns are rejected. Missing counts in
nullable columns are informational and are not violations.

## 5. Dependencies

Runtime (`requirements.txt`): `PyYAML>=6.0`, `pandas>=2.0`, `pydantic>=2.5,<3`.
Dev (`requirements-dev.txt`): runtime plus `pytest>=8.0`. Python >= 3.10.
Later-phase packages are intentionally not listed yet.

Versions actually used for the test run: Python 3.12.3, pandas 3.0.2,
pydantic 2.14.0, PyYAML 6.0.3, pytest 9.1.1. The lower bounds in the
requirements files were not tested.

## 6. Tests run and results

`python -m pytest` -> **62 passed** in about 0.7 s.

| File | Tests | Covers |
|---|---:|---|
| `unit/test_config_loader.py` | 12 | default load, YAML-derived facts, path resolution, pending reporting, missing/invalid/duplicate-key YAML, unknown and missing keys, env override, path existence check |
| `unit/test_config_validation.py` | 20 | target/excluded/batch not features, duplicates, contract-role alignment, availability rules, encoder alignment, ordinal order, min>max, absolute/`..` paths, out-of-range thresholds |
| `unit/test_data_contract.py` | 19 | missing/unexpected columns, nulls, dtypes, integer-valued floats, categories, binary, documented range, no invented ranges, uniqueness, batch values, nulls in nullable columns |
| `unit/test_no_hardcoding.py` | 1 | no column names, categories or paths as literals under `src/` (confirmed it fails when a literal is added) |
| `integration/test_real_data.py` | 10 | real CSVs conform; shapes (4000x11, 1500x12); 3 batches of 500; no Batch in train; target complete; only the 3 documented columns have missing values; categories; ID uniqueness and no overlap |

Also run: `validate-config` (valid, 15 pending listed) and `check-contract`
(train and incoming both conform). An editable install (`pip install -e .`)
was checked to import and run from another directory.

## 7. Files created / modified

All files are new (nothing existed before). Created: every file in the tree in
section 2, including the copied raw CSVs and the brief in `docs/`. The ZIP
contents were copied unchanged.

## 8. How to run Phase 1

```bash
cd mlops-house-price-pipeline
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pip install -e . --no-deps
python -m mlops_pipeline validate-config
python -m mlops_pipeline check-contract
python -m pytest
```
Without installing the package: `PYTHONPATH=src python -m mlops_pipeline ...`
or `make validate-config check-contract test`.

## 9. Known issues and flags

1. **Baseline defaults not from the brief** (marked in the YAML): `random_seed: 42`,
   median imputation, no scaling, ordinal encoding for Property_Condition (order
   taken from the brief's listing), one-hot for City_Zone, `handle_unknown: ignore`.
   Confirm or change them.
2. **15 settings are null on purpose** (section 3). Code that consumes them
   must treat null as "not decided", not as a default.
3. **Age_Years** is documented as int but loads as float64 because of NaN. The
   contract accepts integer-valued floats for nullable integer columns.
4. **Value ranges are largely unspecified.** Only Neighborhood_Score (1 to 10) and
   categorical/binary sets come from the brief. Ranges for the other numeric
   columns are for the data-gate work to decide with justification.
5. **No git repository or DVC yet.** Raw CSVs (about 280 KB) sit in `data/raw/`
   and are not git-ignored. The brief expects data versioning; the choice is
   open. `git init` has not been run.
6. **Placeholder packages** (`pipeline`, `features`, `models`, `serving`,
   `monitoring`) are empty by design.
7. **Descriptive observation, not analysis**: incoming batch means differ from
   train (mean Area_sqft 1150 in train vs 1152 / 1064 / 936 in Batches 1-3;
   mean Price_Lakhs 61.9 vs 62.0 / 63.8 / 69.6). Missing counts are 20 per
   column per batch. The brief says batches may not resemble training data;
   this was not investigated further.
8. Only Python 3.12 / pandas 3 was tested.

## 10. Instructions for continuing with Phase 2

Do not start until the Phase 2 scope is confirmed by the project owner. The
Phase 1 deliverables do not assume what Phase 2 contains.

1. Load everything through `load_config()`; add no hard-coded paths, column
   names or thresholds. `test_no_hardcoding.py` enforces this for dataset
   literals.
2. Any new setting goes into `config/config.yaml` first, then into
   `config/schema.py` (and `validation.py` if it must agree with another
   section), then tests. Unknown keys fail on load, so the schema must be
   updated together with the YAML.
3. Resolve pending settings only with a recorded justification. Replace the
   `null` in the YAML, run `validate-config` (the pending list shrinks) and add
   a test. Do not invent values.
4. Confirm or change the baseline defaults in section 9, item 1.
5. Reuse `DataContract.from_config(...).check(df, "train" | "incoming")` for
   dataset conformance. Gate behaviour (halting, keeping the previous model)
   still has to be built on top of it.
6. Add dependencies to `requirements.txt` only when first used.
7. Keep `python -m pytest` green; extend `tests/unit` and `tests/integration`.
8. Update this handoff, or write `PHASE_2_HANDOFF.md`, at the end of the phase.
