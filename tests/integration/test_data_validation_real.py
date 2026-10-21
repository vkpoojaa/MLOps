"""The data validation layer applied to the actual supplied files (read-only)."""

import hashlib

import pandas as pd
import pytest

from mlops_pipeline.data import DataValidator

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def validator(settings):
    return DataValidator.from_settings(settings)


@pytest.fixture(scope="module")
def train(settings):
    return pd.read_csv(settings.train_path)


@pytest.fixture(scope="module")
def incoming(settings):
    return pd.read_csv(settings.incoming_path)


def test_training_file_passes_the_train_profile(validator, train):
    report = validator.validate(train, "train")
    assert report.passed, report.to_text()
    assert report.n_rows == len(train)


def test_incoming_file_and_each_batch_pass_the_incoming_profile(validator, settings, train, incoming):
    ref = train[settings.config.identifier_column]
    assert validator.validate(incoming, "incoming", reference_ids=ref).passed
    for batch, part in incoming.groupby(settings.config.batch_column):
        report = validator.validate(part, "incoming", reference_ids=ref)
        assert report.passed, f"batch {batch}: {report.to_text()}"


def test_features_of_incoming_data_pass_the_inference_profile(validator, settings, incoming):
    assert validator.validate(incoming[settings.config.feature_columns], "inference").passed


def test_incoming_file_is_not_a_valid_training_file(validator, incoming):
    assert not validator.validate(incoming, "train").passed  # batch metadata must not reach training


def test_provided_files_are_unchanged_by_validation(validator, settings, train, incoming):
    digests = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in (settings.train_path, settings.incoming_path)}
    validator.validate(train, "train")
    validator.validate(incoming, "incoming", reference_ids=train[settings.config.identifier_column])
    assert digests == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in digests}
