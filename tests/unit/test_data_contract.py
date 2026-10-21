import pandas as pd
import pytest

from mlops_pipeline.data import ContractViolationError, DataContract


@pytest.fixture(scope="module")
def contract(settings):
    return DataContract.from_config(settings.config)


def _rules(report):
    return {(v.rule, v.column) for v in report.violations}


def test_valid_train_frame_conforms(contract, valid_train_df):
    report = contract.check(valid_train_df, "train")
    assert report.ok, report.to_text()
    assert report.null_counts["Age_Years"] == 1


def test_valid_incoming_frame_conforms(contract, valid_incoming_df):
    assert contract.check(valid_incoming_df, "incoming").ok


def test_batch_is_unexpected_in_train(contract, valid_incoming_df):
    assert ("unexpected_column", "Batch") in _rules(contract.check(valid_incoming_df, "train"))


def test_batch_is_required_in_incoming(contract, valid_train_df):
    assert ("missing_column", "Batch") in _rules(contract.check(valid_train_df, "incoming"))


def test_missing_column_detected(contract, valid_train_df):
    df = valid_train_df.drop(columns=["City_Zone"])
    assert ("missing_column", "City_Zone") in _rules(contract.check(df, "train"))


def test_unexpected_column_detected(contract, valid_train_df):
    df = valid_train_df.assign(Extra=1)
    assert ("unexpected_column", "Extra") in _rules(contract.check(df, "train"))


def test_nulls_allowed_only_in_nullable_columns(contract, valid_train_df):
    df = valid_train_df.copy()
    df.loc[0, "Price_Lakhs"] = None
    df.loc[1, "Area_sqft"] = None
    rules = _rules(contract.check(df, "train"))
    assert ("null_not_allowed", "Price_Lakhs") in rules
    assert ("null_not_allowed", "Area_sqft") in rules
    assert ("null_not_allowed", "Age_Years") not in rules


def test_wrong_dtype_detected(contract, valid_train_df):
    df = valid_train_df.copy()
    df["Area_sqft"] = ["a", "b", "c"]
    assert ("dtype", "Area_sqft") in _rules(contract.check(df, "train"))


def test_fractional_values_rejected_in_integer_column(contract, valid_train_df):
    df = valid_train_df.copy()
    df.loc[0, "Age_Years"] = 10.5
    assert ("dtype", "Age_Years") in _rules(contract.check(df, "train"))


def test_integer_valued_floats_accepted_for_nullable_integer(contract, valid_train_df):
    assert valid_train_df["Age_Years"].dtype.kind == "f"
    assert ("dtype", "Age_Years") not in _rules(contract.check(valid_train_df, "train"))


def test_unknown_category_detected(contract, valid_train_df):
    df = valid_train_df.copy()
    df.loc[0, "City_Zone"] = "Atlantis"
    assert ("allowed_values", "City_Zone") in _rules(contract.check(df, "train"))


def test_binary_column_allowed_values(contract, valid_train_df):
    df = valid_train_df.copy()
    df.loc[0, "Has_Parking"] = 2
    assert ("allowed_values", "Has_Parking") in _rules(contract.check(df, "train"))


def test_documented_range_enforced(contract, valid_train_df):
    df = valid_train_df.copy()
    df.loc[0, "Neighborhood_Score"] = 11.0
    df.loc[2, "Neighborhood_Score"] = 0.5
    rules = _rules(contract.check(df, "train"))
    assert ("max", "Neighborhood_Score") in rules
    assert ("min", "Neighborhood_Score") in rules


def test_undocumented_ranges_are_not_invented(contract, valid_train_df):
    df = valid_train_df.copy()
    df.loc[0, "Area_sqft"] = 10**7
    assert contract.check(df, "train").ok


def test_duplicate_identifier_detected(contract, valid_train_df):
    df = valid_train_df.copy()
    df.loc[1, "House_ID"] = df.loc[0, "House_ID"]
    assert ("unique", "House_ID") in _rules(contract.check(df, "train"))


def test_batch_values_enforced(contract, valid_incoming_df):
    df = valid_incoming_df.copy()
    df.loc[0, "Batch"] = 4
    assert ("allowed_values", "Batch") in _rules(contract.check(df, "incoming"))


def test_missing_values_in_nullable_columns_are_not_violations(contract, valid_train_df):
    df = valid_train_df.copy()
    df["Age_Years"] = float("nan")
    assert contract.check(df, "train").ok


def test_raise_if_failed(contract, valid_train_df):
    contract.check(valid_train_df, "train").raise_if_failed()
    with pytest.raises(ContractViolationError):
        contract.check(valid_train_df.drop(columns=["Price_Lakhs"]), "train").raise_if_failed()


def test_empty_frame_with_right_columns_conforms(contract, valid_train_df):
    assert contract.check(valid_train_df.iloc[0:0].astype(valid_train_df.dtypes.to_dict()), "train").ok
