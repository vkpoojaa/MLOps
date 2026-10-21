"""Train/validation split and feature/target separation."""

import pandas as pd
import pytest

from mlops_pipeline.data import separate_features_target, split_train_validation


@pytest.fixture
def big_df(valid_train_df):
    """100 synthetic rows with unique ids, built from the conftest fixture's schema."""
    df = pd.concat([valid_train_df] * 34, ignore_index=True).head(100)
    df["House_ID"] = [f"HS-{i}" for i in range(len(df))]
    return df


def test_features_exclude_identifier_batch_and_target(settings, valid_incoming_df):
    cfg = settings.config
    X, y = separate_features_target(valid_incoming_df, cfg)
    assert list(X.columns) == cfg.feature_columns
    assert not set(cfg.excluded_columns) & set(X.columns)
    assert cfg.target not in X.columns
    assert y.name == cfg.target


def test_missing_required_column_raises(settings, valid_train_df):
    with pytest.raises(KeyError):
        separate_features_target(valid_train_df.drop(columns=[settings.config.target]), settings.config)


def test_missing_target_values_rejected(settings, valid_train_df):
    df = valid_train_df.copy()
    df.loc[0, settings.config.target] = None
    with pytest.raises(ValueError, match="target"):
        separate_features_target(df, settings.config)


def test_split_is_disjoint_complete_and_sized(settings, big_df):
    cfg = settings.config
    s = split_train_validation(big_df, cfg)
    assert not set(s.ids_train) & set(s.ids_valid)
    assert len(s.X_train) + len(s.X_valid) == len(big_df)
    assert len(s.X_valid) == round(len(big_df) * cfg.model.split.test_size)
    assert len(s.X_train) == len(s.y_train) and len(s.X_valid) == len(s.y_valid)


def test_split_is_reproducible_and_seed_dependent(settings, big_df):
    cfg = settings.config
    a, b = split_train_validation(big_df, cfg), split_train_validation(big_df, cfg)
    assert list(a.ids_valid) == list(b.ids_valid)
    other = cfg.model_copy(update={"reproducibility": cfg.reproducibility.model_copy(
        update={"random_seed": cfg.random_seed + 1})})
    assert list(split_train_validation(big_df, other).ids_valid) != list(a.ids_valid)


def test_identifiers_never_in_feature_matrices(settings, big_df):
    cfg = settings.config
    s = split_train_validation(big_df, cfg)
    for X in (s.X_train, s.X_valid):
        assert cfg.identifier_column not in X.columns
