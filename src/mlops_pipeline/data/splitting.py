"""Train/validation split and feature/target separation.

Rules enforced here (all column names come from the YAML config):

* split FIRST, so nothing fitted later can ever see validation rows;
* the identifier and batch columns are excluded from ``X`` (they are returned
  separately only so leakage checks can verify the split is disjoint);
* ``X`` contains exactly the configured feature columns, in configured order.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from ..config.schema import AppConfig


@dataclass(frozen=True)
class DataSplit:
    X_train: pd.DataFrame
    X_valid: pd.DataFrame
    y_train: pd.Series
    y_valid: pd.Series
    ids_train: pd.Series  # identifier column: NOT a feature, kept for leakage checks only
    ids_valid: pd.Series

    @property
    def sizes(self) -> dict[str, int]:
        return {"train": len(self.X_train), "validation": len(self.X_valid)}


def separate_features_target(
    df: pd.DataFrame, cfg: AppConfig
) -> tuple[pd.DataFrame, pd.Series]:
    """Return ``(X, y)``; ``X`` holds only the configured feature columns."""
    missing = [c for c in [*cfg.feature_columns, cfg.target] if c not in df.columns]
    if missing:
        raise KeyError(f"columns missing from data: {missing}")
    X = df[cfg.feature_columns].copy()
    y = df[cfg.target].copy()
    forbidden = set(cfg.excluded_columns) | {cfg.target}
    leaked = forbidden & set(X.columns)
    if leaked:  # cannot happen with a valid config; kept as an explicit guard
        raise ValueError(f"forbidden columns in feature matrix: {sorted(leaked)}")
    if y.isna().any():
        raise ValueError("target contains missing values")
    return X, y


def split_train_validation(df: pd.DataFrame, cfg: AppConfig) -> DataSplit:
    """Random, seeded train/validation split of the historical data."""
    train_df, valid_df = train_test_split(
        df,
        test_size=cfg.model.split.test_size,
        random_state=cfg.random_seed,
        shuffle=True,
    )
    X_train, y_train = separate_features_target(train_df, cfg)
    X_valid, y_valid = separate_features_target(valid_df, cfg)
    return DataSplit(
        X_train=X_train,
        X_valid=X_valid,
        y_train=y_train,
        y_valid=y_valid,
        ids_train=train_df[cfg.identifier_column],
        ids_valid=valid_df[cfg.identifier_column],
    )
