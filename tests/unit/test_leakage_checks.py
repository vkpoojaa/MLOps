"""The leakage checks must PASS on a correct run and FAIL on deliberately leaky ones."""

import numpy as np
import pandas as pd
import pytest

from mlops_pipeline.data import split_train_validation
from mlops_pipeline.models.evaluation import evaluate_model
from mlops_pipeline.models.factory import PREPROCESSING_STEP
from mlops_pipeline.models.leakage import check_leakage


@pytest.fixture
def scenario(settings):
    cfg = settings.config
    rng = np.random.default_rng(1)
    n = 200
    df = pd.DataFrame({c: rng.uniform(1, 10, n) for c in cfg.numeric_features})
    for c in cfg.categorical_features:
        allowed = cfg.data_contract.columns[c].allowed_values
        df[c] = [allowed[i % len(allowed)] for i in range(n)]
    df[cfg.numeric_features[3]] = df[cfg.numeric_features[3]].where(df.index % 9 != 0)
    df[cfg.identifier_column] = [f"id{i}" for i in range(n)]
    df[cfg.target] = df[cfg.numeric_features[0]] * 3 + rng.normal(size=n)
    split = split_train_validation(df, cfg)
    results = [evaluate_model(r, cfg.model_spec(r), split, cfg) for r in ("baseline", "candidate")]
    return cfg, df, split, results


def _failed(checks):
    return [c.name for c in checks if not c.passed]


def test_correct_run_passes_all_checks(scenario):
    cfg, df, split, results = scenario
    assert _failed(check_leakage(cfg, split, df, results)) == []


def test_detects_imputer_fitted_on_validation_rows(scenario):
    cfg, df, split, results = scenario
    leaky = results[0]
    everything = pd.concat([split.X_train, split.X_valid])
    leaky.pipeline.named_steps[PREPROCESSING_STEP].fit(everything)  # leak: fit sees validation
    failed = _failed(check_leakage(cfg, split, df, results))
    assert any("imputation_values_equal_training_only" in n for n in failed)


def test_detects_overlapping_split(scenario):
    cfg, df, split, results = scenario
    from dataclasses import replace
    overlapping = replace(split, ids_valid=split.ids_train.iloc[:5])
    assert "train_validation_disjoint_and_complete" in _failed(
        check_leakage(cfg, overlapping, df, results))


def test_detects_identifier_in_feature_matrix(scenario):
    cfg, df, split, results = scenario
    from dataclasses import replace
    bad = replace(split, X_train=split.X_train.assign(**{cfg.identifier_column: "x"}))
    assert "identifier_batch_target_absent_from_feature_matrices" in _failed(
        check_leakage(cfg, bad, df, results))
