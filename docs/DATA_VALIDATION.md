# Data validation layer (Phase 3A)

Module: `src/mlops_pipeline/data/validation.py` - configuration:
`validation:` section of `config/config.yaml` - tests:
`tests/unit/test_data_validation*.py`, `tests/integration/test_data_validation_real.py`.

This layer **only reports**. It does not impute, clip, drop or modify anything, it does
not read files, and it does not halt a run. Enforcing the gate (`validation.on_failure: halt`,
keeping the previous model serving) is a separate step that consumes the report (Part 2).

## How it relates to existing code

| Piece | Role |
|---|---|
| `data/contract.py` (`DataContract`) | Unchanged. Structural conformance used by `load_training_data`. |
| `data/validation.py` (`DataValidator`) | New. Reads the **same** `data_contract` column specs (dtype, nullable, allowed values, uniqueness, brief-documented min/max) and adds the rules the contract does not hold. |
| `features/preprocessing.py` | Unchanged. Validation runs *before* it; preprocessing never validates. |
| `config/validation.py` | Gains `check_data_validation_settings`, run at config load and again inside `DataValidator(...)`. |

## Usage

```python
from mlops_pipeline.config import load_config
from mlops_pipeline.data import DataValidator

settings = load_config()
validator = DataValidator.from_settings(settings)          # ConfigError if the settings are bad
report = validator.validate(df, "incoming", reference_ids=train_df["House_ID"])  # column name: cfg.identifier_column
report.passed            # overall result
report.failures          # tuple[CheckResult]; each has name, status, column, observed, expected, reason
report.to_dict()         # JSON-serialisable (for logs / MLflow artifacts later)
report.to_text()         # explanatory message for a halted run
report.raise_if_failed() # DataValidationError carrying the report
```

`CheckResult.status` is `pass`, `fail` or `skipped`. Skipped checks (disabled in config, not
applicable to the profile, or impossible because a prerequisite failed) are always listed with
a reason and never count as failures. An unexpected exception inside a check group is
converted into a failed `<group>_internal_error` result (fail closed).

## Profiles (training vs inference)

| Profile | Required roles | Optional roles | Minimum rows | Duplicate-record check | IDs disjoint from training |
|---|---|---|---:|:-:|:-:|
| `train` | identifier, target, features | - | 1000 | yes | no |
| `incoming` | identifier, batch, target, features | - | 100 | yes | yes |
| `inference` | features | identifier, batch, target | 1 | no | no |

Anything outside required + optional is an `unexpected_columns` failure (the contract's
`allow_unexpected_columns: false`). So `Batch` in a training file fails, and a scoring request
without a price is valid under `inference` but not under `train`. Optional columns that are
present are validated like any other.

## Checks

| Group (`validation.checks.*`) | Check name | Column | Fails when |
|---|---|---|---|
| `schema` | `duplicate_column_names` | - | a column label occurs twice |
| | `row_count` | - | rows < profile `min_rows` |
| | `column_present` / `target_present` | each required column | required column absent |
| | `unexpected_columns` | each extra column | column not in the profile |
| | `dtype` | each present column | contract dtype not met (text vs numeric, integer-valued, nullable-int-as-float allowed) |
| `missing_values` | `missing_fraction` | each non-target column | fraction > per-column limit (nullable columns: from config; non-nullable: 0) |
| | `missing_fraction_overall` | - | missing cells / all validated cells > `max_overall_fraction` |
| `value_ranges` | `non_finite` | numeric columns | any +/-inf (NaN is "missing", not "non-finite") |
| | `numeric_range` | numeric columns | value outside the inclusive effective range |
| `categorical_values` | `allowed_values` | columns with `allowed_values` | value outside the set (case-sensitive; missing is not "invalid") |
| `target` | `target_not_missing`, `target_finite`, `target_range`, `target_not_constant` | target | missing / inf / out of range / fewer than `min_distinct_values` distinct values |
| `duplicates` | `duplicate_records` | - | duplicated records (compare_roles columns, identifier ignored) > `max_duplicate_fraction` |
| `identifiers` | `unique_values` | every column with `unique: true` | duplicate values |
| | `identifier_format` | identifier | not a full match of `identifiers.pattern` |
| | `identifier_disjoint_from_training` | identifier | overlap with `reference_ids`; **fails** if the profile requires it and no `reference_ids` were passed |

Effective numeric range = tightest of the contract's `min/max` (brief-documented) and
`validation.numeric_ranges`. Dependent checks are skipped (not double-reported) when the column
is absent or not numeric/text as required.

## Configuration keys (all under `validation:` in `config/config.yaml`)

`checks.{schema,value_ranges,missing_values,categorical_values,target,duplicates,identifiers}` -
`missing_values.max_overall_fraction` - `missing_values.max_fraction_per_column.<column>` -
`numeric_ranges.<column>.{min,max}` - `target.min_distinct_values` -
`duplicates.{compare_roles,max_duplicate_fraction}` - `identifiers.pattern` -
`profiles.<train|incoming|inference>.{contract_dataset,required_roles,optional_roles,min_rows,check_duplicate_records,identifier_disjoint_from_training}` -
`on_failure` (only `halt`).

### Configuration is validated, and fails safely

Load-time (`load_config` -> `ConfigError`, all problems listed together) and again in
`DataValidator(...)`: missing/null limits, a nullable column without a limit, a numeric column
without a complete `[min, max]`, ranges for unknown / non-numeric / `allowed_values` columns,
ranges wider than the contract's documented range, `min > max`, a positive limit for a
non-nullable column, `max_overall_fraction: 0` together with positive column limits, a profile
requiring a column absent from its contract dataset, a `train` profile that does not require the
target, disjointness for `train` or without an identifier role, an invalid regex, dependent checks
enabled while `schema` is off, unknown keys, out-of-range fractions.

## Threshold rationale and assumptions

**All values below are proposals for the project owner to confirm.** The brief documents only:
the column dictionary and types, `Neighborhood_Score` in 1-10, the category sets, `Has_Parking`
in {0,1}, unique `House_ID`, `Batch` in {1,2,3} for incoming only, and that three columns contain
missing values. Nothing else here comes from the brief; every other number was derived from the
supplied files (observed values were measured, not copied from an earlier session).

| Setting | Value | Evidence and reasoning |
|---|---|---|
| per-column missing limit (`Age_Years`, `Distance_to_Center_km`, `Neighborhood_Score`) | 0.08 | Observed exactly 4.0% in each column in train (160/4000) **and** in every 500-row batch (20/500). 8% = 2x observed = 4.5 binomial sd above 4% for n=500 (false-alarm probability about 1.6e-5), yet small enough to catch a real degradation. Caveat: for a 100-row batch the same limit has about a 2% false-alarm chance. |
| `max_overall_fraction` | 0.015 | Observed 1.09% (train), 1.00% (each batch). 1.5% is 1.4x observed (3.9 sd above the expected count for a batch). It is intentionally tighter than three columns all at their 8% limit (about 2.2%), so it binds when several columns degrade together (three at 6% = 1.5%). |
| non-nullable columns | 0 | From `data_contract` (`nullable: false`); no config key needed. |
| `Area_sqft` | 100 - 6000 | Observed 350-2699. Wide on purpose: catches zero/negative values and gross scale errors (e.g. a x10 error) without halting a legitimately different batch. |
| `Bedrooms`, `Bathrooms` | 1 - 10 | Observed 1-6 and 1-5. Training has no 0-bedroom rows, so a 0 would be an extrapolation the model has no basis for; set min to 0 if studios should be accepted. |
| `Age_Years` | 0 - 100 | Observed 0-40 (looks capped at 40). A negative age is impossible. |
| `Distance_to_Center_km` | 0 - 100 | Observed 1-45. |
| `Neighborhood_Score` | 1 - 10 | From the brief, kept in `data_contract` only (no duplicate in `numeric_ranges`). |
| `Price_Lakhs` (target) | 1 - 500 | Observed 8-157.9. Must be positive; 500 is about 3x the observed maximum. |
| `target.min_distinct_values` | 2 | Logical minimum: a constant target carries no signal. Real data has thousands of distinct values. |
| `max_duplicate_fraction` | 0.0 | Observed 0 duplicates in train and incoming (also when ignoring the id). Copies under new ids would land in both CV folds and the validation split and inflate scores. Not applied to `inference` (identical requests are legitimate). |
| `identifiers.pattern` | `^HS-[0-9]{6}$` | Observed in both files (`HS-000001`...`HS-004000`, `HS-101000`...`HS-103499`). **Not in the brief**; catches blanks, whitespace and case changes. Set to `null` to disable. |
| `min_rows` train / incoming / inference | 1000 / 100 / 1 | 25% of 4000 and 20% of 500: detects truncated files without tying the gate to exact counts (the brief's row counts describe the supplied files, not a requirement). |
| incoming `identifier_disjoint_from_training` | true | Observed 0 overlap. An incoming id already in training would bias drift and performance monitoring. |

Ranges are corruption/plausibility guards, **not drift detection**. The brief says incoming
batches may differ from the training data; that is Task 7. Observed batch maxima (e.g.
`Area_sqft` max 2445 / 2152 in batches 2 / 3) lie well inside the ranges.

Other assumptions: missing values are NaN/empty (blank CSV cells); string comparison of
categories is exact and case-sensitive (stricter than the preprocessor, which silently ignores
unseen categories - that is intended, the gate should stop before that happens); the `pending`
list reported by `validate-config` no longer contains any `validation.*` entry.

## Not covered (deliberately)

Distribution shift and per-batch comparison with training (monitoring phase), cross-column
logic (e.g. bathrooms vs bedrooms - the brief gives no rule), temporal checks (no date column),
row-level missing-count limits (23 training rows have >= 2 of 3 nullable values missing; no
brief requirement, imputation handles them), and acting on a failure.

## Intended pipeline integration point (Part 2, not implemented here)

The brief's "validate data" stage between *ingest* and *build features*. Concretely, in
`models/training.py::run_training`, call `validator.validate(df, "train").raise_if_failed()` on
the DataFrame returned by `load_training_data`, before `split_train_validation`; for each new
batch, `validate(batch, "incoming", reference_ids=training ids)` before it is used for
monitoring or retraining; and for serving, `validate(payload, "inference")` at the API boundary.
The gate wraps the report: log `report.to_dict()`, halt on `not report.passed`, leave the
production model untouched.
