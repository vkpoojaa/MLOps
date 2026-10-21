"""End-to-end Phase 2 run on the real training file (outputs go to a temp dir)."""

import dataclasses
import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor

from mlops_pipeline.models import run_training
from mlops_pipeline.models.evaluation import compute_metrics

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def sandbox(settings, tmp_path_factory):
    """Settings whose project root is a temp dir holding a copy of the training CSV only."""
    root = tmp_path_factory.mktemp("proj")
    target = root / settings.config.paths.train_data
    target.parent.mkdir(parents=True)
    shutil.copy(settings.train_path, target)
    return dataclasses.replace(settings, project_root=root)


@pytest.fixture(scope="module")
def reads(sandbox):
    """Run training once, recording every CSV path pandas is asked to read."""
    opened: list[Path] = []
    original = pd.read_csv

    def spy(path, *a, **k):
        opened.append(Path(path).resolve())
        return original(path, *a, **k)

    mp = pytest.MonkeyPatch()
    mp.setattr(pd, "read_csv", spy)
    try:
        run = run_training(sandbox, save=True)
    finally:
        mp.undo()
    return run, opened


@pytest.fixture(scope="module")
def run(reads):
    return reads[0]


def test_only_the_training_file_is_read_never_incoming(sandbox, settings, reads):
    _, opened = reads
    assert opened == [sandbox.train_path]
    assert settings.incoming_path not in opened
    assert not sandbox.incoming_path.exists()  # the sandbox does not even contain it


def test_split_sizes(run, settings):
    n = 4000
    assert run.split.sizes == {"validation": round(n * settings.config.model.split.test_size),
                               "train": n - round(n * settings.config.model.split.test_size)}


def test_all_leakage_checks_pass(run):
    failed = [c for c in run.leakage if not c.passed]
    assert not failed, failed
    assert len(run.leakage) >= 8


def test_forbidden_columns_never_in_model_inputs(run, settings):
    cfg = settings.config
    for r in run.results.values():
        seen = set(r.pipeline.named_steps["preprocess"].feature_names_in_)
        assert seen == set(cfg.feature_columns)
        assert not seen & {*cfg.excluded_columns, cfg.target}


def test_predictions_ignore_identifier_and_batch_columns(run, settings):
    cfg = settings.config
    raw = pd.read_csv(settings.train_path).head(200)
    with_meta = raw.assign(**{cfg.batch_column: 2}).drop(columns=[cfg.target])
    features_only = raw[cfg.feature_columns]
    for r in run.results.values():
        np.testing.assert_allclose(r.pipeline.predict(with_meta), r.pipeline.predict(features_only))


def test_both_models_beat_a_mean_predictor_on_validation(run, settings):
    cfg = settings.config
    dummy = DummyRegressor().fit(run.split.X_train, run.split.y_train)
    base = compute_metrics(run.split.y_valid, dummy.predict(run.split.X_valid), cfg.metric_names)
    for r in run.results.values():
        assert r.validation_metrics["mae"] < base["mae"]
        assert r.validation_metrics["rmse"] < base["rmse"]
        assert r.validation_metrics["r2"] > base["r2"] + 0.5


def test_both_models_use_identical_evaluation_methodology(run, settings):
    cfg = settings.config
    for r in run.results.values():
        assert list(r.validation_metrics) == cfg.metric_names
        assert list(r.train_metrics) == cfg.metric_names
        assert list(r.cv_metrics) == cfg.metric_names
        assert len(r.cv_metrics["mae"]["per_fold"]) == cfg.model.evaluation.cv_folds
    assert run.results["baseline"].n_features_after_preprocessing == \
        run.results["candidate"].n_features_after_preprocessing


def test_training_is_reproducible(sandbox, run):
    again = run_training(sandbox, save=False)
    for role in run.results:
        assert again.results[role].validation_metrics == run.results[role].validation_metrics
        assert again.results[role].cv_metrics == run.results[role].cv_metrics


def test_reference_validation_metrics(run):
    """Values observed on the real data in Phase 2 (seed 42); loose tolerance across versions."""
    ref = {"baseline": {"mae": 5.7931, "rmse": 7.5044, "r2": 0.8831},
           "candidate": {"mae": 5.7777, "rmse": 7.5846, "r2": 0.8806}}
    for role, expected in ref.items():
        for metric, value in expected.items():
            assert run.results[role].validation_metrics[metric] == pytest.approx(value, abs=0.05)


def test_training_candidates_never_replace_production_pipeline(run, sandbox, settings):
    # Phase 6: the legacy training API may save candidate artifacts, but only a
    # future explicit promotion stage may write selected_model.joblib.
    production_path = sandbox.resolve(settings.config.paths.selected_model)
    assert not production_path.exists()
    assert "selected" not in run.saved_paths
    assert run.report["selection"]["production_model_replaced"] is False


def test_report_is_written_and_consistent(run, sandbox, settings):
    cfg = settings.config
    report = json.loads(sandbox.resolve(cfg.paths.evaluation_report).read_text())
    assert report["settings"]["random_seed"] == cfg.random_seed
    assert report["settings"]["excluded_columns"] == cfg.excluded_columns
    assert report["selection"]["selected_in_config"] == cfg.model.selected
    assert all(c["passed"] for c in report["leakage_checks"])
    for role in ("baseline", "candidate"):
        assert report["models"][role]["validation_metrics"] == run.results[role].validation_metrics
    assert report["data"]["n_train"] + report["data"]["n_validation"] == report["data"]["n_rows"]
