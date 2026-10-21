"""Preprocessing: missing values, encoding, reproducibility, no-leakage behaviour."""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from mlops_pipeline.features import build_preprocessor


@pytest.fixture
def X(settings):
    """Small feature frame (config-driven columns) with gaps and varied categories."""
    cfg = settings.config
    n = 12
    rng = np.random.default_rng(0)
    data = {c: rng.uniform(1, 10, n) for c in cfg.numeric_features}
    enc = cfg.preprocessing.categorical.encoders
    for c in cfg.categorical_features:
        allowed = cfg.data_contract.columns[c].allowed_values
        data[c] = [allowed[i % len(allowed)] for i in range(n)]
    df = pd.DataFrame(data)
    df.loc[1, cfg.numeric_features[3]] = np.nan
    df.loc[4, cfg.numeric_features[4]] = np.nan
    return df


def test_output_is_complete_numeric_dataframe_without_nans(settings, X):
    out = build_preprocessor(settings.config).fit_transform(X)
    assert isinstance(out, pd.DataFrame)
    assert not out.isna().any().any()
    assert all(pd.api.types.is_numeric_dtype(t) for t in out.dtypes)


def test_median_imputation_uses_fitted_training_values(settings, X):
    cfg = settings.config
    col = cfg.numeric_features[3]
    pre = build_preprocessor(cfg).fit(X)
    new = X.head(2).copy()
    new[col] = np.nan
    out = pre.transform(new)
    assert out[f"numeric__{col}"].tolist() == [X[col].median()] * 2


def test_statistics_come_only_from_fit_data(settings, X):
    cfg = settings.config
    col = cfg.numeric_features[3]
    pre = build_preprocessor(cfg).fit(X.iloc[:6])
    other = X.iloc[6:].copy()
    other[col] = np.nan
    out = pre.transform(other)  # transform must not re-learn anything
    assert (out[f"numeric__{col}"] == X.iloc[:6][col].median()).all()


def test_ordinal_encoding_follows_configured_order(settings, X):
    cfg = settings.config
    ordinal_cols = [c for c, e in cfg.preprocessing.categorical.encoders.items() if e.type == "ordinal"]
    assert ordinal_cols, "config is expected to define one ordinal encoder"
    col, order = ordinal_cols[0], cfg.preprocessing.categorical.encoders[ordinal_cols[0]].order
    rows = pd.concat([X.iloc[[0]].assign(**{col: v}) for v in order], ignore_index=True)
    out = build_preprocessor(cfg).fit(X).transform(rows)
    assert out[f"cat_{col}__{col}"].tolist() == list(range(len(order)))


def test_onehot_columns_and_unknown_category_ignored(settings, X):
    cfg = settings.config
    onehot = [c for c, e in cfg.preprocessing.categorical.encoders.items() if e.type == "onehot"][0]
    pre = build_preprocessor(cfg).fit(X)
    unseen = X.head(1).assign(**{onehot: "never_seen"})
    out = pre.transform(unseen)
    cols = [c for c in out.columns if c.startswith(f"cat_{onehot}__")]
    assert len(cols) == X[onehot].nunique()
    assert out[cols].to_numpy().sum() == 0  # unknown -> all zeros


def test_unknown_ordinal_category_is_imputed_not_crashing(settings, X):
    cfg = settings.config
    ordinal = [c for c, e in cfg.preprocessing.categorical.encoders.items() if e.type == "ordinal"][0]
    out = build_preprocessor(cfg).fit(X).transform(X.head(1).assign(**{ordinal: "never_seen"}))
    assert not out.isna().any().any()


def test_unknown_categories_raise_when_configured(settings, X):
    cfg = settings.config
    strict = cfg.model_copy(update={"preprocessing": cfg.preprocessing.model_copy(update={
        "categorical": cfg.preprocessing.categorical.model_copy(update={"handle_unknown": "error"})})})
    pre = build_preprocessor(strict).fit(X)
    onehot = [c for c, e in cfg.preprocessing.categorical.encoders.items() if e.type == "onehot"][0]
    with pytest.raises(ValueError):
        pre.transform(X.head(1).assign(**{onehot: "never_seen"}))


def test_excluded_and_target_columns_are_dropped_even_if_passed(settings, X):
    cfg = settings.config
    extra = X.assign(**{cfg.identifier_column: "id", cfg.batch_column: 1, cfg.target: 5.0})
    pre = build_preprocessor(cfg).fit(extra)
    names = " ".join(pre.get_feature_names_out())
    for forbidden in (*cfg.excluded_columns, cfg.target):
        assert forbidden not in names
    pd.testing.assert_frame_equal(pre.transform(extra), pre.transform(X))


def test_preprocessing_is_deterministic_and_clone_is_unfitted(settings, X):
    cfg = settings.config
    a = build_preprocessor(cfg).fit_transform(X)
    b = build_preprocessor(cfg).fit_transform(X)
    pd.testing.assert_frame_equal(a, b)
    fitted = build_preprocessor(cfg).fit(X)
    assert not hasattr(clone(fitted), "transformers_")  # cloning discards fitted state


def test_scaling_options_follow_config(settings, X):
    cfg = settings.config
    scaled = cfg.model_copy(update={"preprocessing": cfg.preprocessing.model_copy(update={
        "numeric": cfg.preprocessing.numeric.model_copy(update={"scaling": "standard"})})})
    out = build_preprocessor(scaled).fit_transform(X)
    col = f"numeric__{cfg.numeric_features[0]}"
    assert abs(out[col].mean()) < 1e-9 and abs(out[col].std(ddof=0) - 1) < 1e-9


def test_missing_indicator_option_adds_columns(settings, X):
    cfg = settings.config
    ind = cfg.model_copy(update={"preprocessing": cfg.preprocessing.model_copy(update={
        "numeric": cfg.preprocessing.numeric.model_copy(update={
            "imputation": cfg.preprocessing.numeric.imputation.model_copy(
                update={"add_missing_indicator": True})})})})
    base = build_preprocessor(settings.config).fit_transform(X)
    with_ind = build_preprocessor(ind).fit_transform(X)
    assert with_ind.shape[1] == base.shape[1] + 2  # the two columns that had gaps
