"""Evaluation module: one methodology for every model.

Metrics: MAE, RMSE (both in the target's unit, lower is better) and R²
(higher is better). Every model is evaluated the same way:

1. fit on the training split only,
2. K-fold cross-validation on the training split only (the pipeline is cloned and
   re-fitted inside each fold, so preprocessing never sees the held-out fold),
3. one-off scoring on the untouched validation split,
4. training-split scores, to expose over-fitting.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

from ..config.schema import AppConfig, ModelSpec
from ..data.splitting import DataSplit
from .factory import PREPROCESSING_STEP, build_pipeline


def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


METRIC_FUNCTIONS = {
    "mae": lambda y, p: float(mean_absolute_error(y, p)),
    "rmse": _rmse,
    "r2": lambda y, p: float(r2_score(y, p)),
}
HIGHER_IS_BETTER = {"mae": False, "rmse": False, "r2": True}


def compute_metrics(y_true, y_pred, metric_names: list[str]) -> dict[str, float]:
    unknown = [m for m in metric_names if m not in METRIC_FUNCTIONS]
    if unknown:
        raise ValueError(f"unknown metrics: {unknown}")
    return {m: METRIC_FUNCTIONS[m](y_true, y_pred) for m in metric_names}


def is_better(metric: str, value: float, than: float) -> bool:
    """True if ``value`` is strictly better than ``than`` for this metric."""
    return value > than if HIGHER_IS_BETTER[metric] else value < than


def cross_validate(
    pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, cfg: AppConfig
) -> dict[str, dict[str, float]]:
    """K-fold CV; returns ``{metric: {"mean", "std", "per_fold"}}`` over folds."""
    folds = KFold(n_splits=cfg.model.evaluation.cv_folds, shuffle=True, random_state=cfg.random_seed)
    per_fold: dict[str, list[float]] = {m: [] for m in cfg.metric_names}
    for fit_idx, hold_idx in folds.split(X):
        model = clone(pipeline)  # fresh, unfitted: statistics are learned on the fold's training part only
        model.fit(X.iloc[fit_idx], y.iloc[fit_idx])
        scores = compute_metrics(y.iloc[hold_idx], model.predict(X.iloc[hold_idx]), cfg.metric_names)
        for name, value in scores.items():
            per_fold[name].append(value)
    return {
        name: {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)), "per_fold": v}
        for name, v in per_fold.items()
    }


@dataclass
class ModelResult:
    role: str
    spec: ModelSpec
    pipeline: Pipeline  # fitted on the training split only
    train_metrics: dict[str, float]
    validation_metrics: dict[str, float]
    cv_metrics: dict[str, dict[str, float]]
    n_features_after_preprocessing: int
    extras: dict = field(default_factory=dict)


def evaluate_model(role: str, spec: ModelSpec, split: DataSplit, cfg: AppConfig) -> ModelResult:
    """Fit ``spec`` on the training split and score it with the shared methodology."""
    pipeline = build_pipeline(cfg, spec)
    cv = cross_validate(pipeline, split.X_train, split.y_train, cfg)

    pipeline.fit(split.X_train, split.y_train)  # only fit on the final training split
    names = cfg.metric_names
    return ModelResult(
        role=role,
        spec=spec,
        pipeline=pipeline,
        train_metrics=compute_metrics(split.y_train, pipeline.predict(split.X_train), names),
        validation_metrics=compute_metrics(split.y_valid, pipeline.predict(split.X_valid), names),
        cv_metrics=cv,
        n_features_after_preprocessing=len(pipeline.named_steps[PREPROCESSING_STEP].get_feature_names_out()),
    )


def best_by_primary_metric(results: list[ModelResult], cfg: AppConfig) -> ModelResult:
    primary = cfg.model.evaluation.primary_metric
    best = results[0]
    for r in results[1:]:
        if is_better(primary, r.validation_metrics[primary], best.validation_metrics[primary]):
            best = r
    return best
