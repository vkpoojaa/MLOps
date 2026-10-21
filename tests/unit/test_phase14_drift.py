import hashlib
from pathlib import Path

import joblib
import pytest

from mlops_pipeline.config.loader import load_config
from mlops_pipeline.models.training import run_training
from mlops_pipeline.monitoring.drift import monitor_incoming_batches


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def phase14_result(tmp_path_factory):
    settings = load_config(check_paths=True)
    output = tmp_path_factory.mktemp("phase14")

    # Generate a deterministic test model if the production artifact is absent.
    # Do not replace an existing production model.
    model_path = settings.resolve(settings.config.paths.selected_model)
    if not model_path.is_file():
        model_path.parent.mkdir(parents=True, exist_ok=True)
        training_run = run_training(settings, save=False)
        selected_role = settings.config.model.selected
        joblib.dump(training_run.results[selected_role].pipeline, model_path)

    return (
        settings,
        _sha256(settings.incoming_path),
        monitor_incoming_batches(settings, output_dir=output),
    )


def test_phase14_supplied_batches_are_unchanged_and_results_are_evidence_based(
    phase14_result,
):
    settings, before, result = phase14_result
    after = _sha256(settings.incoming_path)

    assert before == after
    assert result.report["methodology"]["incoming_unchanged"] is True
    assert [row["batch"] for row in result.report["batches"]] == [1, 2, 3]


def test_phase14_actual_triggered_batches_match_supplied_data(phase14_result):
    _, _, result = phase14_result
    rows = {row["batch"]: row for row in result.report["batches"]}

    assert rows[1]["data_drift_trigger"] is False
    assert rows[1]["performance_drift_trigger"] is False
    assert rows[2]["data_drift_trigger"] is True
    assert rows[2]["performance_drift_trigger"] is True
    assert rows[3]["data_drift_trigger"] is True
    assert rows[3]["performance_drift_trigger"] is True

    assert rows[2]["drifted_features"] == [
        "Area_sqft",
        "Bedrooms",
        "Bathrooms",
        "Property_Condition",
        "City_Zone",
    ]
    assert rows[3]["drifted_features"] == rows[2]["drifted_features"]


def test_phase14_outputs_are_reproducible(phase14_result):
    _, _, result = phase14_result
    methodology = result.report["methodology"]

    assert methodology["per_feature_thresholds"]["Area_sqft"] == pytest.approx(
        0.034434312122830386
    )
    assert methodology["data_drift_share_threshold"] == pytest.approx(2 / 9)
    assert methodology[
        "performance_relative_mae_degradation_threshold"
    ] == pytest.approx(0.06028119854325451)


def test_phase14_calibration_and_operational_thresholds_are_configured(settings):
    cfg = settings.config.monitoring

    assert cfg.data_drift.bootstrap_runs == 1000
    assert cfg.data_drift.bootstrap_batch_size == 500
    assert cfg.performance_drift.bootstrap_runs == 2000
    assert cfg.performance_drift.bootstrap_batch_size == 500
    assert cfg.operational.latency_p95_ms_max == pytest.approx(100.0)
    assert cfg.operational.error_rate_max == pytest.approx(0.01)
