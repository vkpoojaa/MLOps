from __future__ import annotations

from mlops_pipeline.config import load_config
from mlops_pipeline.models.quality_gate import evaluate_model_gate


def _production(mae=5.70):
    return {
        "model_name": "current-production",
        "artifact_path": "artifacts/models/selected_model.joblib",
        "metrics": {"mae": mae, "rmse": 7.4, "r2": 0.89},
    }


def test_candidate_passes_both_independent_checks(settings):
    decision = evaluate_model_gate({"mae": 5.60, "rmse": 7.2, "r2": 0.90}, settings.config, _production())
    assert decision["status"] == "PASS"
    assert decision["absolute_quality_check"]["status"] == "PASS"
    assert decision["production_comparison"]["status"] == "PASS"
    assert decision["promotion_performed"] is False
    assert decision["production_model_preserved"] is True


def test_candidate_fails_absolute_threshold(settings):
    decision = evaluate_model_gate({"mae": 8.0, "rmse": 10.0, "r2": 0.50}, settings.config, _production())
    assert decision["status"] == "FAIL"
    assert decision["absolute_quality_check"]["status"] == "FAIL"
    assert "absolute threshold" in decision["failure_reasons"][0]
    assert decision["candidate_metrics"]["mae"] == 8.0
    assert decision["production_comparison"]["production_metrics"]["mae"] == 5.70
    assert decision["candidate_disposition"] == "rejected"
    assert decision["production_model_replaced"] is False


def test_candidate_passes_absolute_but_is_worse_than_production(settings):
    # 5.75 <= the evidence-derived absolute MAE threshold 5.793117..., but
    # 5.75 is worse than this controlled production model's MAE of 5.70.
    decision = evaluate_model_gate({"mae": 5.75, "rmse": 7.5, "r2": 0.88}, settings.config, _production(mae=5.70))
    assert decision["absolute_quality_check"]["status"] == "PASS"
    assert decision["production_comparison"]["status"] == "FAIL"
    assert decision["status"] == "FAIL"
    assert decision["production_comparison"]["production_value"] == 5.70
    assert decision["production_model_preserved"] is True


def test_first_model_case_requires_absolute_quality_only(settings):
    decision = evaluate_model_gate({"mae": 5.70, "rmse": 7.2, "r2": 0.90}, settings.config, None)
    assert decision["status"] == "PASS"
    assert decision["absolute_quality_check"]["status"] == "PASS"
    assert decision["production_comparison"]["status"] == "SKIPPED"
    assert "First-model case" in decision["production_comparison"]["reason"]


def test_controlled_poor_model_is_rejected(settings):
    # Deterministic poor-model demonstration: deliberately supplied bad metrics,
    # not a manipulated production artifact or production metric.
    poor_metrics = {"mae": 50.0, "rmse": 60.0, "r2": -2.0}
    decision = evaluate_model_gate(poor_metrics, settings.config, _production())
    assert decision["status"] == "FAIL"
    assert decision["candidate_disposition"] == "rejected"
    assert decision["candidate_metrics"] == poor_metrics
    assert decision["production_model_preserved"] is True
