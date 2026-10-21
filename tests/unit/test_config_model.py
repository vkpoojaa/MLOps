"""Phase 2 model/evaluation section of the config."""

import pytest

from mlops_pipeline.config import ConfigError, load_config
from mlops_pipeline.models.factory import ESTIMATORS


def _fails(write_config, mutate, match):
    with pytest.raises(ConfigError, match=match):
        load_config(write_config(mutate))


def test_real_config_defines_baseline_candidate_and_metrics(settings):
    m = settings.config.model
    assert m.baseline.name != m.candidate.name
    assert m.evaluation.primary_metric == "mae"
    assert settings.config.metric_names == ["mae", "rmse", "r2"]
    assert 0 < m.split.test_size < 1
    assert settings.config.random_seed == 42


def test_algorithms_in_schema_match_factory_registry():
    from typing import get_args
    from mlops_pipeline.config.schema import Algorithm
    assert set(get_args(Algorithm)) == set(ESTIMATORS)


def test_selected_may_be_null_meaning_pending(write_config):
    s = load_config(write_config(lambda d: d["model"].update(selected=None)))
    assert s.config.model.selected is None
    assert "model.selected" in s.pending


@pytest.mark.parametrize("bad", ["champion", "Baseline", 1])
def test_selected_must_be_baseline_or_candidate(write_config, bad):
    _fails(write_config, lambda d: d["model"].update(selected=bad), "selected")


def test_unknown_algorithm_rejected(write_config):
    _fails(write_config, lambda d: d["model"]["candidate"].update(algorithm="deep_magic"), "algorithm")


def test_unknown_metric_rejected(write_config):
    _fails(write_config, lambda d: d["model"]["evaluation"].update(primary_metric="accuracy"),
           "primary_metric")


def test_primary_metric_must_not_repeat_in_secondary(write_config):
    _fails(write_config, lambda d: d["model"]["evaluation"].update(secondary_metrics=["mae", "r2"]),
           "must not repeat")


@pytest.mark.parametrize("role", ["baseline", "candidate"])
def test_random_state_must_not_be_in_hyperparameters(write_config, role):
    _fails(write_config,
           lambda d: d["model"][role]["hyperparameters"].update(random_state=1),
           "do not set random_state")


def test_baseline_and_candidate_names_must_differ(write_config):
    def mutate(d):
        d["model"]["candidate"]["name"] = d["model"]["baseline"]["name"]
    _fails(write_config, mutate, "different names")


@pytest.mark.parametrize("folds", [0, 1])
def test_cv_folds_at_least_two(write_config, folds):
    _fails(write_config, lambda d: d["model"]["evaluation"].update(cv_folds=folds), "cv_folds")


@pytest.mark.parametrize("size", [0, 1.0, None])
def test_test_size_must_be_fraction(write_config, size):
    _fails(write_config, lambda d: d["model"]["split"].update(test_size=size), "test_size")


def test_phase2_decisions_are_recorded_and_no_longer_pending(settings):
    pending = set(settings.pending)
    resolved = {"model.baseline", "model.candidate", "model.split.test_size",
                "model.evaluation.primary_metric", "model.algorithm", "model.hyperparameters"}
    assert not (pending & resolved)
    assert settings.config.model.selected in {"baseline", "candidate"}
    # the selected role must be a configured model whose pipeline is saved at the configured path
    assert settings.config.paths.selected_model.endswith(".joblib")
