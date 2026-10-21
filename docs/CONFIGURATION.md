# Configuration Guide

## Source of truth

`config/config.yaml` is the single configuration source for the implemented pipeline. Executable code reads validated settings from the typed configuration layer rather than maintaining independent copies of paths, model settings, gate thresholds, or monitoring calibration parameters.

Validate before execution:

```bash
PYTHONPATH=src python -m mlops_pipeline validate-config
```

## Main sections

| Section | Controls |
|---|---|
| `project` | project identity and configured paths |
| `data` / `data_contract` | raw inputs, columns, roles, dtypes, allowed values |
| `preprocessing` | numeric imputation/scaling and categorical encoders |
| `model` | baseline/candidate algorithms, fixed hyperparameters, split and metrics |
| `validation` | schema, range, missingness, categorical, target, duplicate, identifier checks |
| `gates` | absolute Model Gate and production-comparison policy |
| `mlflow` | tracking URI, registered-model name, production alias and tracking metadata |
| `drift` | reference data, PSI calibration, bootstrap settings and performance-degradation threshold |
| `monitoring` | operational metric/alert settings |
| `reproducibility` | random seed and reproducibility controls |

## Changing configuration safely

1. Edit `config/config.yaml` only.
2. Run `validate-config`.
3. Run the relevant unit/integration tests.
4. Run the affected stage or complete pipeline.
5. Review the resulting evidence before changing production.

Do not duplicate configuration values in Python modules or notebooks.

## Thresholds

Validation limits, Model Gate thresholds, drift calibration controls, and operational settings are documented beside their YAML definitions where the value is evidence-derived. The Phase 14 drift thresholds are calibrated from the supplied training reference rather than selected by visual inspection.

## Reproducibility

The random seed is configured under `reproducibility`. Model hyperparameters are configured under `model`; the candidate settings are not tuned at runtime. The reproducibility checker can be run with:

```bash
PYTHONPATH=src python -m mlops_pipeline reproducibility-check
```

## Configuration failure behavior

The typed configuration loader raises configuration errors for missing/invalid values and cross-section inconsistencies. A configuration failure should be fixed before running the pipeline; it is not silently defaulted to an unrelated hard-coded value.
