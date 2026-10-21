from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from mlops_pipeline.config import load_config
from mlops_pipeline.data import DataValidator
from mlops_pipeline.monitoring import mlflow_tracking


class _FakeMLflow:
    def __init__(self, events):
        self.events = events
        self.info = SimpleNamespace(run_id="fake-run-123")
        self.sklearn = SimpleNamespace(log_model=lambda model, artifact_path: events.append(("model", artifact_path)))

    @contextmanager
    def start_run(self, **kwargs):
        self.events.append(("start_run", kwargs))
        yield SimpleNamespace(info=self.info)
        self.events.append(("end_run", kwargs))

    def set_tags(self, tags): self.events.append(("tags", tags))
    def set_tag(self, key, value): self.events.append(("tag", key, value))
    def log_param(self, key, value): self.events.append(("param", key, value))
    def log_metrics(self, metrics): self.events.append(("metrics", metrics))
    def log_artifact(self, path, artifact_path=None): self.events.append(("artifact", Path(path).name, artifact_path))


def test_pipeline_creates_mlflow_run_before_fit_and_logs_lineage(tmp_path, monkeypatch):
    project = Path(__file__).resolve().parents[2]
    settings = load_config(project / "config" / "config.yaml", check_paths=True)
    frame = pd.read_csv(settings.train_path)
    report = DataValidator.from_settings(settings).validate(frame, "train")
    assert report.passed
    events = []
    fake = _FakeMLflow(events)
    monkeypatch.setattr(mlflow_tracking, "load_mlflow", lambda _settings: fake)

    class Spec:
        name = "test-model"
        algorithm = "linear_regression"
        hyperparameters = {"fit_intercept": True}

    fitted = SimpleNamespace(
        validation_metrics={"mae": 1.2, "rmse": 1.5, "r2": 0.9},
        pipeline=object(), extras={},
    )
    def fit():
        events.append(("fit",))
        return fitted

    result, run_id = mlflow_tracking.fit_and_track_model(
        settings=settings, role="baseline", spec=Spec(), dataset_path=settings.train_path,
        validation_report=report, fit_callable=fit,
    )
    assert result is fitted
    assert run_id == "fake-run-123"
    assert events.index(next(e for e in events if e[0] == "start_run")) < events.index(("fit",))
    assert any(e[0] == "param" and e[1] == "model.type" and e[2] == "linear_regression" for e in events)
    assert any(e[0] == "metrics" and set(e[1]) == {"MAE", "RMSE", "R2"} for e in events)
    assert any(e[0] == "tags" and "dataset.sha256" in e[1] and "code.version" in e[1] for e in events)
    assert any(e[0] == "artifact" and e[1] == "config.yaml" for e in events)
    assert fitted.extras["mlflow_run_id"] == run_id


def test_tracking_enabled_without_mlflow_fails_clearly(monkeypatch):
    settings = load_config(Path(__file__).resolve().parents[2] / "config" / "config.yaml")
    def unavailable_import(_settings):
        raise RuntimeError("MLflow tracking is enabled in config.yaml but MLflow is not installed")
    monkeypatch.setattr(mlflow_tracking, "load_mlflow", unavailable_import)
    with pytest.raises(RuntimeError, match="MLflow tracking is enabled"):
        mlflow_tracking.fit_and_track_model(
            settings=settings, role="baseline", spec=SimpleNamespace(name="x", algorithm="linear_regression"),
            dataset_path=settings.train_path, validation_report=SimpleNamespace(passed=True),
            fit_callable=lambda: pytest.fail("fit must not run if tracking cannot initialize"),
        )
