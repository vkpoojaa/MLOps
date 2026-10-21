"""Integration tests proving the training quality gate halts before side effects."""
from __future__ import annotations

import pandas as pd
import pytest

import mlops_pipeline.models.training as training
from mlops_pipeline.data import DataValidationError


def test_bad_data_blocks_training_and_artifact_side_effects(settings, monkeypatch):
    # Deterministic bad-data fixture in memory; supplied CSVs are never modified.
    df = pd.read_csv(settings.train_path).copy(deep=True)
    df.loc[df.index[0], "Area_sqft"] = -999
    monkeypatch.setattr(training, "load_training_data", lambda _settings: df)

    called = []
    def forbidden(*args, **kwargs):
        called.append((args, kwargs))
        raise AssertionError("downstream function must not be called after gate failure")

    monkeypatch.setattr(training, "split_train_validation", forbidden)
    monkeypatch.setattr(training, "evaluate_model", forbidden)
    monkeypatch.setattr(training.joblib, "dump", forbidden)

    with pytest.raises(DataValidationError) as caught:
        training.run_training(settings, save=True)

    report = caught.value.report
    assert not report.passed
    assert any(check.name == "numeric_range" and check.column == "Area_sqft"
               for check in report.failures)
    assert report.to_dict()["passed"] is False
    assert called == []


def test_validation_configuration_is_checked_before_downstream_work(settings, monkeypatch):
    # A malformed AppConfig cannot be constructed through normal config loading;
    # simulate validator construction failure to verify fail-closed behavior.
    monkeypatch.setattr(training, "load_training_data", lambda _settings: pd.read_csv(settings.train_path))
    def fail_build(_settings):
        raise ValueError("simulated invalid validation configuration")
    monkeypatch.setattr(training.DataValidator, "from_settings", fail_build)
    called = []
    monkeypatch.setattr(training, "split_train_validation", lambda *a, **k: called.append("split"))
    with pytest.raises(ValueError, match="invalid validation configuration"):
        training.run_training(settings, save=True)
    assert called == []
