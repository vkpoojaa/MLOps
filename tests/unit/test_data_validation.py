"""Unit tests for the data validation layer.

Every threshold, range, category and column name used here is derived from the
loaded YAML configuration (nothing is hard-coded), so changing the config
changes what the tests construct. Valid data is a copy of the supplied training
file; each test then breaks exactly one thing in a copy.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest
from pandas.api import types as pdt

from mlops_pipeline.data import (CheckStatus, DataValidationError, DataValidator, ValidationReport,
                                 validate_dataframe)

TRAIN, INCOMING, INFERENCE = "train", "incoming", "inference"


# ------------------------------------------------------------------ fixtures
@pytest.fixture(scope="module")
def cfg(settings):
    return settings.config


@pytest.fixture(scope="module")
def validator(cfg):
    return DataValidator(cfg)


@pytest.fixture(scope="module")
def _train_raw(settings):
    return pd.read_csv(settings.train_path)


@pytest.fixture(scope="module")
def _incoming_raw(settings):
    return pd.read_csv(settings.incoming_path)


@pytest.fixture
def train_df(_train_raw):
    return _train_raw.copy()


@pytest.fixture
def incoming_df(_incoming_raw):
    return _incoming_raw.copy()


# ------------------------------------------------------------------- helpers
def _spec(cfg, col):
    return cfg.data_contract.columns[col]


def _range(cfg, col):
    """Effective [lo, hi] = contract bounds combined with validation.numeric_ranges."""
    s, r = _spec(cfg, col), cfg.validation.numeric_ranges.get(col)
    lows = [x for x in (s.min, r.min if r else None) if x is not None]
    highs = [x for x in (s.max, r.max if r else None) if x is not None]
    return max(lows), min(highs)


def _range_columns(cfg):
    """Feature columns that get a numeric-range check (no allowed_values) plus the target."""
    cols = [c for c in cfg.numeric_features if _spec(cfg, c).allowed_values is None]
    return cols, cfg.target


def _nullable(cfg):
    return [c for c in cfg.feature_columns if _spec(cfg, c).nullable]


def _non_nullable_features(cfg):
    return [c for c in cfg.feature_columns if not _spec(cfg, c).nullable]


def _set(df, col, value, rows=(0,)):
    """Assign without changing an integer column's dtype when the value is integral."""
    if pdt.is_integer_dtype(df[col]) and float(value).is_integer():
        value = int(value)
    df.loc[list(rows), col] = value


def _fail_names(report: ValidationReport) -> set[tuple[str, str | None]]:
    return {(r.name, r.column) for r in report.failures}


# --------------------------------------------------------------- valid data
@pytest.mark.parametrize("profile,frame", [(TRAIN, "train_df"), (INCOMING, "incoming_df")])
def test_valid_data_passes_with_complete_structured_results(request, validator, train_df, profile, frame):
    df = request.getfixturevalue(frame)
    ids = train_df[validator._cfg.identifier_column]
    report = validator.validate(df, profile, reference_ids=ids)
    assert report.passed, report.to_text()
    assert report.failures == ()
    assert report.counts["fail"] == 0 and report.counts["pass"] > 0
    for r in report.results:
        assert r.name and isinstance(r.status, CheckStatus)
        assert (r.reason is None) == (r.status is CheckStatus.PASS)  # reasons only on fail/skip
    assert json.loads(json.dumps(report.to_dict()))["passed"] is True  # JSON serialisable
    report.raise_if_failed()  # must not raise


def test_every_incoming_batch_passes_on_its_own(validator, cfg, incoming_df, train_df):
    for _, batch in incoming_df.groupby(cfg.batch_column):
        report = validator.validate(batch, INCOMING, reference_ids=train_df[cfg.identifier_column])
        assert report.passed, report.to_text()


def test_checks_cover_every_requirement_family(validator, train_df):
    names = {r.name for r in validator.validate(train_df, TRAIN).results}
    expected = {"duplicate_column_names", "row_count", "column_present", "target_present", "unexpected_columns",
                "dtype", "missing_fraction", "missing_fraction_overall", "non_finite", "numeric_range",
                "allowed_values", "target_not_missing", "target_finite", "target_range", "target_not_constant",
                "duplicate_records", "unique_values", "identifier_format"}
    assert expected <= names


def test_validation_never_modifies_the_supplied_dataframe(validator, train_df):
    before = train_df.copy(deep=True)
    dtypes, columns, index = train_df.dtypes.copy(), list(train_df.columns), train_df.index.copy()
    validator.validate(train_df, TRAIN)
    validator.validate(train_df, INFERENCE)
    pd.testing.assert_frame_equal(train_df, before)
    assert list(train_df.columns) == columns and train_df.index.equals(index)
    assert (train_df.dtypes == dtypes).all()


def test_supplied_csv_files_are_not_written(settings, validator, train_df):
    path = settings.train_path
    before = (path.stat().st_mtime_ns, path.read_bytes())
    validator.validate(train_df, TRAIN)
    assert (path.stat().st_mtime_ns, path.read_bytes()) == before


# --------------------------------------------------------- required columns
def test_missing_required_column_fails_for_each_required_role(validator, cfg, train_df):
    for col in [*cfg.feature_columns, cfg.target, cfg.identifier_column]:
        report = validator.validate(train_df.drop(columns=[col]), TRAIN)
        name = "target_present" if col == cfg.target else "column_present"
        assert not report.passed
        r = report.get(name, col)
        assert r.failed and r.column == col and r.observed is False and r.expected is True
        assert col in r.reason


def test_missing_target_skips_target_checks_instead_of_crashing(validator, cfg, train_df):
    report = validator.validate(train_df.drop(columns=[cfg.target]), TRAIN)
    assert report.get("target_present").failed
    assert report.get("target", cfg.target).status is CheckStatus.SKIPPED


def test_unexpected_column_fails_unless_contract_allows_it(validator, cfg, train_df):
    train_df["totally_new_column"] = 1
    report = validator.validate(train_df, TRAIN)
    assert cfg.data_contract.allow_unexpected_columns is False
    assert ("unexpected_columns", "totally_new_column") in _fail_names(report)


def test_batch_column_is_rejected_in_training_but_required_in_incoming(validator, cfg, train_df, incoming_df):
    assert not validator.validate(incoming_df, TRAIN).passed
    assert ("unexpected_columns", cfg.batch_column) in _fail_names(validator.validate(incoming_df, TRAIN))
    without_batch = incoming_df.drop(columns=[cfg.batch_column])
    report = validator.validate(without_batch, INCOMING, reference_ids=train_df[cfg.identifier_column])
    assert ("column_present", cfg.batch_column) in _fail_names(report)


def test_duplicate_column_names_fail_and_do_not_crash(validator, train_df):
    dup = pd.concat([train_df, train_df.iloc[:, [1]]], axis=1)
    report = validator.validate(dup, TRAIN)
    assert report.get("duplicate_column_names").failed
    assert not [r for r in report.results if r.name.endswith("_internal_error")]


# ------------------------------------------------------------- schema/types
def test_wrong_dtype_fails_and_dependent_value_checks_are_skipped(validator, cfg, train_df):
    col = cfg.numeric_features[0]
    train_df[col] = train_df[col].astype(object)
    train_df.loc[0, col] = "not a number"
    report = validator.validate(train_df, TRAIN)
    assert report.get("dtype", col).failed
    assert report.get("numeric_range", col).status is CheckStatus.SKIPPED
    assert not report.passed


def test_text_in_target_fails_dtype_and_target_checks_skip(validator, cfg, train_df):
    train_df[cfg.target] = train_df[cfg.target].astype(str)
    report = validator.validate(train_df, TRAIN)
    assert report.get("dtype", cfg.target).failed
    assert report.get("target_range").status is CheckStatus.SKIPPED
    assert not report.passed


def test_row_count_below_profile_minimum_fails(validator, cfg, train_df):
    minimum = cfg.validation.profiles[TRAIN].min_rows
    report = validator.validate(train_df.head(minimum - 1), TRAIN)
    r = report.get("row_count")
    assert r.failed and r.observed == minimum - 1 and r.expected == {"min": minimum}


def test_empty_dataframe_fails_cleanly(validator, train_df):
    report = validator.validate(train_df.iloc[0:0], TRAIN)
    assert not report.passed and report.get("row_count").failed
    assert not [r for r in report.results if r.name.endswith("_internal_error")]


# ----------------------------------------------------------- numeric values
def test_value_above_and_below_range_fails_per_column(validator, cfg, train_df):
    cols, target = _range_columns(cfg)
    for col in cols:
        lo, hi = _range(cfg, col)
        for bad in (hi + 1, lo - 1):
            df = train_df.copy()
            _set(df, col, bad)
            report = validator.validate(df, TRAIN)
            r = report.get("numeric_range", col)
            assert r.failed, (col, bad)
            assert r.expected == {"min": lo, "max": hi}
            assert r.observed["n_out_of_range"] == 1
            assert not report.passed
        # exactly at the bounds is allowed (inclusive)
        df = train_df.copy()
        _set(df, col, lo, rows=(0,))
        _set(df, col, hi, rows=(1,))
        assert validator.validate(df, TRAIN).get("numeric_range", col).passed


def test_infinite_values_fail_for_every_float_column(validator, cfg, train_df):
    cols, target = _range_columns(cfg)
    float_cols = [c for c in [*cols, target] if pdt.is_float_dtype(train_df[c])]
    assert float_cols
    for col in float_cols:
        for bad in (np.inf, -np.inf):
            df = train_df.copy()
            df.loc[0, col] = bad
            report = validator.validate(df, TRAIN)
            name = "target_finite" if col == cfg.target else "non_finite"
            assert report.get(name, col).failed, (col, bad)
            assert not report.passed


def test_numeric_value_outside_contract_documented_range_fails(validator, cfg, train_df):
    documented = [c for c in cfg.numeric_features if _spec(cfg, c).min is not None]
    assert documented, "the brief documents at least one range"
    col = documented[0]
    df = train_df.copy()
    df.loc[0, col] = _spec(cfg, col).max + 1
    assert validator.validate(df, TRAIN).get("numeric_range", col).failed


# ------------------------------------------------------------------- target
def test_target_missing_values_fail(validator, cfg, train_df):
    train_df.loc[[0, 1], cfg.target] = np.nan
    r = validator.validate(train_df, TRAIN).get("target_not_missing")
    assert r.failed and r.observed == 2 and r.expected == 0 and r.column == cfg.target


def test_target_out_of_range_fails_both_sides(validator, cfg, train_df):
    lo, hi = _range(cfg, cfg.target)
    for bad in (hi + 1, lo - 1, 0, -1):
        df = train_df.copy()
        df.loc[0, cfg.target] = bad
        expect_fail = bad < lo or bad > hi
        assert validator.validate(df, TRAIN).get("target_range").failed is expect_fail


def test_constant_target_fails(validator, cfg, train_df):
    lo, hi = _range(cfg, cfg.target)
    train_df[cfg.target] = (lo + hi) / 2
    r = validator.validate(train_df, TRAIN).get("target_not_constant")
    assert r.failed and r.observed == 1 and r.expected == {"min": cfg.validation.target.min_distinct_values}


# --------------------------------------------------------------- missingness
def test_excessive_missingness_in_each_nullable_column_fails(validator, cfg, train_df):
    limits = cfg.validation.missing_values.max_fraction_per_column
    for col in _nullable(cfg):
        df = train_df.copy()
        n_missing = math.floor(limits[col] * len(df)) + 1
        df.loc[df.index[:n_missing], col] = np.nan
        r = validator.validate(df, TRAIN).get("missing_fraction", col)
        assert r.failed and r.expected == {"max": limits[col]} and r.observed > limits[col]


def test_missingness_exactly_at_the_limit_passes(validator, cfg, train_df):
    limits = cfg.validation.missing_values.max_fraction_per_column
    col = _nullable(cfg)[0]
    df = train_df.copy()
    n_missing = math.floor(limits[col] * len(df))
    df[col] = df[col].astype(float)
    df.loc[:, col] = df[col].fillna(df[col].median())
    df.loc[df.index[:n_missing], col] = np.nan
    r = validator.validate(df, TRAIN).get("missing_fraction", col)
    assert r.observed <= limits[col] and r.passed


def test_any_missing_value_in_a_non_nullable_column_fails(validator, cfg, train_df):
    for col in _non_nullable_features(cfg):
        df = train_df.copy()
        df[col] = df[col].astype(object) if not pdt.is_float_dtype(df[col]) else df[col]
        df.loc[0, col] = np.nan
        report = validator.validate(df, TRAIN)
        r = report.get("missing_fraction", col)
        assert r.failed and r.expected == {"max": 0.0}, col


def test_overall_missing_limit_can_fail_when_every_column_limit_passes(write_config, settings, train_df):
    observed = float(train_df.isna().to_numpy().mean())
    tight = observed / 2  # strictly below what the real data has, still >= 0
    path = write_config(lambda d: d["validation"]["missing_values"].update(max_overall_fraction=tight))
    from mlops_pipeline.config import load_config
    strict = DataValidator(load_config(path).config)
    report = strict.validate(train_df, TRAIN)
    assert report.get("missing_fraction_overall").failed
    assert report.get("missing_fraction_overall").expected == {"max": tight}
    assert all(r.passed for r in report.results if r.name == "missing_fraction")


# --------------------------------------------------------------- categories
def test_invalid_category_fails_for_every_constrained_column(validator, cfg, train_df, incoming_df):
    checked = 0
    for df, profile in ((train_df, TRAIN), (incoming_df, INCOMING)):
        for col in [c for c in df.columns if _spec(cfg, c).allowed_values is not None]:
            allowed = _spec(cfg, col).allowed_values
            bad = "not-a-valid-category" if isinstance(allowed[0], str) else max(allowed) + 1000
            mutated = df.copy()
            if isinstance(bad, str):
                mutated[col] = mutated[col].astype(object)
            mutated.loc[mutated.index[0], col] = bad
            report = validator.validate(mutated, profile, reference_ids=train_df[cfg.identifier_column])
            r = report.get("allowed_values", col)
            assert r.failed, col
            assert r.expected == list(allowed)
            assert r.observed["n_invalid"] == 1 and str(bad) in r.observed["examples"]
            checked += 1
    assert checked >= len(cfg.categorical_features)


def test_wrong_case_category_is_invalid(validator, cfg, train_df):
    col = cfg.categorical_features[0]
    train_df.loc[0, col] = str(train_df.loc[0, col]).lower()
    assert validator.validate(train_df, TRAIN).get("allowed_values", col).failed


def test_missing_categorical_is_not_reported_as_invalid_category(validator, cfg, train_df):
    col = cfg.categorical_features[0]
    train_df.loc[0, col] = np.nan
    report = validator.validate(train_df, TRAIN)
    assert report.get("allowed_values", col).passed      # missing is the missing-value check's job
    assert report.get("missing_fraction", col).failed


# ------------------------------------------------- identifiers & duplicates
def test_duplicate_identifier_fails(validator, cfg, train_df):
    ident = cfg.identifier_column
    train_df.loc[1, ident] = train_df.loc[0, ident]
    r = validator.validate(train_df, TRAIN).get("unique_values", ident)
    assert r.failed and r.observed == 1 and r.expected == 0


def test_malformed_identifier_fails_when_a_pattern_is_configured(validator, cfg, train_df):
    assert cfg.validation.identifiers.pattern is not None
    ident = cfg.identifier_column
    train_df.loc[0, ident] = " "
    r = validator.validate(train_df, TRAIN).get("identifier_format", ident)
    assert r.failed and r.expected == cfg.validation.identifiers.pattern and r.observed["n_invalid"] == 1


def test_duplicate_records_under_new_identifiers_fail(validator, cfg, train_df):
    ident = cfg.identifier_column
    donor_id = train_df.loc[1, ident]
    train_df.loc[1] = train_df.loc[0]          # same content as row 0 ...
    train_df.loc[1, ident] = donor_id          # ... but a distinct, still-unique identifier
    report = validator.validate(train_df, TRAIN)
    r = report.get("duplicate_records")
    assert r.failed and r.expected == {"max": cfg.validation.duplicates.max_duplicate_fraction}
    assert r.observed == pytest.approx(1 / len(train_df))
    assert report.get("unique_values", ident).passed  # distinct ids: only the content check catches it


def test_overlap_with_training_identifiers_fails_for_incoming(validator, cfg, train_df, incoming_df):
    ident = cfg.identifier_column
    leaked = incoming_df.copy()
    leaked.loc[0, ident] = train_df.loc[0, ident]
    report = validator.validate(leaked, INCOMING, reference_ids=train_df[ident])
    r = report.get("identifier_disjoint_from_training")
    assert r.failed and r.observed["n_overlap"] == 1 and r.expected == 0


def test_missing_reference_ids_fail_closed_rather_than_skip(validator, cfg, incoming_df):
    r = validator.validate(incoming_df, INCOMING).get("identifier_disjoint_from_training")
    assert r.failed and "not supplied" in r.reason


# ------------------------------------------------------------ profiles
def test_inference_profile_needs_only_the_features(validator, cfg, incoming_df):
    features_only = incoming_df[cfg.feature_columns]
    report = validator.validate(features_only, INFERENCE)
    assert report.passed, report.to_text()
    assert report.get("target", cfg.target).status is CheckStatus.SKIPPED
    # the very same payload is NOT a valid training file
    assert not validator.validate(features_only, TRAIN).passed


def test_inference_profile_still_validates_the_features(validator, cfg, incoming_df):
    payload = incoming_df[cfg.feature_columns].copy()
    col = cfg.categorical_features[0]
    payload[col] = payload[col].astype(object)
    payload.loc[payload.index[0], col] = "not-a-valid-category"
    assert not validator.validate(payload, INFERENCE).passed
    assert not validator.validate(payload.drop(columns=[cfg.feature_columns[0]]), INFERENCE).passed


def test_inference_profile_validates_optional_columns_when_present(validator, cfg, incoming_df):
    lo, hi = _range(cfg, cfg.target)
    payload = incoming_df.copy()
    payload.loc[payload.index[0], cfg.target] = hi + 1
    assert validator.validate(payload, INFERENCE).get("target_range").failed


def test_identical_scoring_requests_are_not_duplicate_failures(validator, cfg, incoming_df):
    payload = pd.concat([incoming_df[cfg.feature_columns].head(1)] * 3, ignore_index=True)
    report = validator.validate(payload, INFERENCE)
    assert report.passed and report.get("duplicate_records").status is CheckStatus.SKIPPED


# ----------------------------------------------- toggles / report / safety
def test_disabled_check_is_reported_as_skipped_and_does_not_fail(write_config, train_df, cfg):
    from mlops_pipeline.config import load_config

    def off(d):
        for key in d["validation"]["checks"]:
            d["validation"]["checks"][key] = False

    validator = DataValidator(load_config(write_config(off)).config)
    train_df.loc[0, cfg.target] = np.nan  # would fail if the target checks were on
    report = validator.validate(train_df, TRAIN)
    assert report.counts["fail"] == 0
    assert {r.name for r in report.results} == set(cfg.validation.checks.model_dump(by_alias=True))
    assert all(r.status is CheckStatus.SKIPPED and "disabled" in r.reason for r in report.results)


def test_failed_report_text_dict_and_raise(validator, cfg, train_df):
    col = cfg.categorical_features[0]
    train_df.loc[0, col] = "not-a-valid-category"
    report = validator.validate(train_df, TRAIN)
    text = report.to_text()
    assert "FAILED" in text and col in text and "allowed_values" in text
    as_dict = report.to_dict()
    assert as_dict["passed"] is False and as_dict["profile"] == TRAIN
    failing = [c for c in as_dict["checks"] if c["status"] == "fail"]
    assert failing and all(set(c) == {"name", "status", "column", "observed", "expected", "reason"} for c in failing)
    assert all(c["reason"] for c in failing)
    with pytest.raises(DataValidationError) as exc:
        report.raise_if_failed()
    assert exc.value.report is report


def test_crashing_check_group_fails_closed(monkeypatch, validator, train_df):
    def boom(self, ctx):
        raise RuntimeError("boom")
    monkeypatch.setattr(DataValidator, "_ranges", boom)
    report = validator.validate(train_df, TRAIN)
    assert not report.passed
    r = report.get("value_ranges_internal_error")
    assert r.failed and "boom" in r.reason


def test_bad_call_arguments_are_rejected(validator, train_df):
    with pytest.raises(TypeError):
        validator.validate([1, 2, 3], TRAIN)
    with pytest.raises(ValueError, match="unknown validation profile"):
        validator.validate(train_df, "staging")
    with pytest.raises(TypeError):
        validator.validate(train_df, TRAIN, reference_ids=train_df)


def test_validate_dataframe_helper_matches_validator(cfg, validator, train_df):
    assert validate_dataframe(train_df, cfg, TRAIN).to_dict() == validator.validate(train_df, TRAIN).to_dict()
