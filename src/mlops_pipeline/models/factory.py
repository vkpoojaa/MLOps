"""Create estimators and the complete (preprocessing + model) pipeline."""

from __future__ import annotations

from typing import Callable

from sklearn.base import RegressorMixin
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline

from ..config.schema import AppConfig, ModelSpec
from ..features import build_preprocessor

# Must stay in sync with ``config.schema.Algorithm`` (enforced by a unit test).
ESTIMATORS: dict[str, Callable[..., RegressorMixin]] = {
    "linear_regression": LinearRegression,
    "ridge": Ridge,
    "random_forest": RandomForestRegressor,
    "gradient_boosting": GradientBoostingRegressor,
}

PREPROCESSING_STEP = "preprocess"
MODEL_STEP = "model"


def build_estimator(spec: ModelSpec, seed: int) -> RegressorMixin:
    """Instantiate the estimator; the seed is injected here, never read from hyperparameters."""
    factory = ESTIMATORS[spec.algorithm]
    params = dict(spec.hyperparameters)
    if "random_state" in factory().get_params():
        params["random_state"] = seed
    try:
        return factory(**params)
    except TypeError as exc:
        raise ValueError(
            f"invalid hyperparameters for {spec.name!r} ({spec.algorithm}): {exc}"
        ) from exc


def build_pipeline(cfg: AppConfig, spec: ModelSpec) -> Pipeline:
    """Unfitted pipeline: config-driven preprocessing followed by the estimator."""
    return Pipeline(
        [
            (PREPROCESSING_STEP, build_preprocessor(cfg)),
            (MODEL_STEP, build_estimator(spec, cfg.random_seed)),
        ]
    )
