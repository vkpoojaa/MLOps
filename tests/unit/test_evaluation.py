"""Evaluation module: metric correctness and CV hygiene."""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from mlops_pipeline.models.evaluation import (
    best_by_primary_metric, compute_metrics, cross_validate, is_better)


def test_metrics_match_hand_computed_values():
    y = np.array([10.0, 20.0, 30.0, 40.0])
    p = np.array([12.0, 18.0, 33.0, 40.0])
    m = compute_metrics(y, p, ["mae", "rmse", "r2"])
    errors = np.array([2, -2, 3, 0.0])
    assert m["mae"] == pytest.approx(7 / 4)
    assert m["rmse"] == pytest.approx(np.sqrt((4 + 4 + 9) / 4))
    ss_res, ss_tot = (errors ** 2).sum(), ((y - y.mean()) ** 2).sum()
    assert m["r2"] == pytest.approx(1 - ss_res / ss_tot)


def test_perfect_prediction_and_mean_predictor():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert compute_metrics(y, y, ["mae", "rmse", "r2"]) == {"mae": 0.0, "rmse": 0.0, "r2": 1.0}
    assert compute_metrics(y, np.full(4, y.mean()), ["r2"])["r2"] == pytest.approx(0.0)


def test_unknown_metric_rejected():
    with pytest.raises(ValueError, match="unknown metrics"):
        compute_metrics([1.0], [1.0], ["accuracy"])


@pytest.mark.parametrize("metric,a,b,expected", [
    ("mae", 1.0, 2.0, True), ("mae", 2.0, 1.0, False), ("mae", 1.0, 1.0, False),
    ("rmse", 1.0, 2.0, True), ("r2", 0.9, 0.8, True), ("r2", 0.8, 0.9, False)])
def test_is_better_respects_metric_direction(metric, a, b, expected):
    assert is_better(metric, a, b) is expected


class _RecordingScaler(BaseEstimator, TransformerMixin):
    """Records the rows seen in fit, to prove each CV fit excludes the held-out fold."""
    seen_fits: list = []

    def fit(self, X, y=None):
        type(self).seen_fits.append(set(X.index))
        return self

    def transform(self, X):
        return X


def test_cv_refits_whole_pipeline_on_each_fold_without_heldout_rows(settings):
    cfg = settings.config
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=100), "b": rng.normal(size=100)})
    y = pd.Series(X["a"] * 2 + rng.normal(scale=0.1, size=100))
    _RecordingScaler.seen_fits = []
    pipe = Pipeline([("rec", _RecordingScaler()), ("m", LinearRegression())])
    out = cross_validate(pipe, X, y, cfg)

    n_folds = cfg.model.evaluation.cv_folds
    assert len(_RecordingScaler.seen_fits) == n_folds
    all_idx = set(X.index)
    held_out = [all_idx - s for s in _RecordingScaler.seen_fits]
    assert all(len(h) == 100 // n_folds for h in held_out)
    assert set.union(*held_out) == all_idx                      # folds cover the data
    assert sum(len(h) for h in held_out) == 100                 # ...without overlap
    assert set(out) == set(cfg.metric_names)
    assert all(len(v["per_fold"]) == n_folds for v in out.values())
    assert out["mae"]["mean"] == pytest.approx(np.mean(out["mae"]["per_fold"]))


def test_cv_does_not_mutate_the_passed_pipeline(settings):
    X = pd.DataFrame({"a": np.arange(50.0)})
    y = pd.Series(np.arange(50.0))
    pipe = Pipeline([("m", LinearRegression())])
    cross_validate(pipe, X, y, settings.config)
    assert not hasattr(pipe.named_steps["m"], "coef_")


def test_best_by_primary_metric_uses_config_metric(settings):
    class R:  # minimal stand-in with the attributes used
        def __init__(self, role, mae): self.role, self.validation_metrics = role, {"mae": mae}
    best = best_by_primary_metric([R("baseline", 5.0), R("candidate", 4.0)], settings.config)
    assert best.role == "candidate"
    assert best_by_primary_metric([R("baseline", 4.0), R("candidate", 4.0)], settings.config).role == "baseline"
