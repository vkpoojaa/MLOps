"""Explicit, machine-checkable data-leakage checks for a trained comparison."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from ..config.schema import AppConfig
from ..data.splitting import DataSplit
from .factory import PREPROCESSING_STEP
from .evaluation import ModelResult
from ..features.preprocessing import NUMERIC_BLOCK


@dataclass(frozen=True)
class LeakageCheck:
    name: str
    passed: bool
    detail: str

    def to_dict(self) -> dict:
        return asdict(self)


def _expected_statistic(frame: pd.DataFrame, strategy: str) -> pd.Series:
    if strategy == "mean":
        return frame.mean()
    if strategy == "median":
        return frame.median()
    return frame.mode().iloc[0]  # most_frequent


def check_leakage(
    cfg: AppConfig, split: DataSplit, full_df: pd.DataFrame, results: list[ModelResult]
) -> list[LeakageCheck]:
    checks: list[LeakageCheck] = []
    forbidden = [*cfg.excluded_columns, cfg.target]

    # 1. forbidden columns never reach the model
    in_matrices = sorted(set(forbidden) & (set(split.X_train.columns) | set(split.X_valid.columns)))
    checks.append(LeakageCheck(
        "identifier_batch_target_absent_from_feature_matrices", not in_matrices,
        f"found: {in_matrices}" if in_matrices else f"none of {forbidden} in X_train/X_valid"))

    for r in results:
        pre: ColumnTransformer = r.pipeline.named_steps[PREPROCESSING_STEP]
        seen_in = set(pre.feature_names_in_)
        out_names = list(pre.get_feature_names_out())
        bad_in = sorted(seen_in & set(forbidden))
        bad_out = [n for n in out_names if any(n.endswith(f) for f in forbidden)]
        checks.append(LeakageCheck(
            f"{r.role}: fitted_pipeline_inputs_and_outputs_exclude_forbidden_columns",
            not bad_in and not bad_out,
            f"inputs={sorted(seen_in)}; forbidden found in inputs={bad_in}, outputs={bad_out}"))

    # 2. split is disjoint and complete
    overlap = set(split.ids_train) & set(split.ids_valid)
    complete = len(split.ids_train) + len(split.ids_valid) == len(full_df)
    checks.append(LeakageCheck(
        "train_validation_disjoint_and_complete", not overlap and complete,
        f"overlap={len(overlap)} ids; train={len(split.ids_train)} + validation="
        f"{len(split.ids_valid)} of {len(full_df)} rows"))

    # 3 & 4. fitted preprocessing statistics come from the training split only
    strategy = cfg.preprocessing.numeric.imputation.strategy
    num_cols = cfg.numeric_features
    expected_train = _expected_statistic(split.X_train[num_cols], strategy)
    expected_all = _expected_statistic(
        pd.concat([split.X_train, split.X_valid])[num_cols], strategy)
    would_differ = [c for c in num_cols if abs(expected_train[c] - expected_all[c]) > 1e-12]
    for r in results:
        pre = r.pipeline.named_steps[PREPROCESSING_STEP]
        fitted = pd.Series(pre.named_transformers_[NUMERIC_BLOCK].named_steps["impute"].statistics_,
                           index=num_cols)
        ok = bool(((fitted - expected_train).abs() < 1e-12).all())
        checks.append(LeakageCheck(
            f"{r.role}: imputation_values_equal_training_only_{strategy}", ok,
            f"fitted values match {strategy} of the training split; sensitivity: using "
            f"train+validation would have changed {len(would_differ)} of {len(num_cols)} "
            f"values, so this check can detect leakage"))
        for column in cfg.categorical_features:
            block = pre.named_transformers_[f"cat_{column}"]
            if isinstance(block, OneHotEncoder):
                learned = sorted(map(str, block.categories_[0]))
                train_cats = sorted(map(str, split.X_train[column].dropna().unique()))
                checks.append(LeakageCheck(
                    f"{r.role}: onehot_categories_learned_from_training_only[{column}]",
                    learned == train_cats, f"learned={learned}"))
    return checks
