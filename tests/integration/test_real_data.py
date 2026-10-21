"""Facts about the actual provided datasets (read from data/raw)."""

import pandas as pd
import pytest

from mlops_pipeline.data import DataContract

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def train(settings):
    return pd.read_csv(settings.train_path)


@pytest.fixture(scope="module")
def incoming(settings):
    return pd.read_csv(settings.incoming_path)


@pytest.fixture(scope="module")
def contract(settings):
    return DataContract.from_config(settings.config)


def test_train_conforms_to_contract(contract, train):
    report = contract.check(train, "train")
    assert report.ok, report.to_text()


def test_incoming_conforms_to_contract(contract, incoming):
    report = contract.check(incoming, "incoming")
    assert report.ok, report.to_text()


def test_dataset_shapes_match_brief(train, incoming):
    assert train.shape == (4000, 11)
    assert incoming.shape == (1500, 12)


def test_incoming_has_three_batches_of_500(settings, incoming):
    counts = incoming[settings.config.batch_column].value_counts().to_dict()
    assert counts == {1: 500, 2: 500, 3: 500}
    assert sorted(counts) == settings.config.batch_values


def test_train_has_no_batch_column(settings, train):
    assert settings.config.batch_column not in train.columns


def test_target_present_and_complete(settings, train, incoming):
    t = settings.config.target
    assert train[t].notna().all() and incoming[t].notna().all()


def test_only_documented_columns_have_missing_values(settings, train, incoming):
    expected = {"Age_Years", "Distance_to_Center_km", "Neighborhood_Score"}
    for df in (train, incoming):
        assert set(df.columns[df.isna().any()]) == expected


def test_features_are_present_in_both_files(settings, train, incoming):
    for df in (train, incoming):
        assert set(settings.config.feature_columns) <= set(df.columns)


def test_categorical_values_match_contract(settings, train, incoming):
    spec = settings.config.data_contract.columns
    for col in settings.config.categorical_features:
        for df in (train, incoming):
            assert set(df[col].unique()) <= set(spec[col].allowed_values)


def test_identifier_is_unique_and_ids_do_not_overlap(settings, train, incoming):
    ident = settings.config.identifier_column
    assert train[ident].is_unique and incoming[ident].is_unique
    assert not set(train[ident]) & set(incoming[ident])
