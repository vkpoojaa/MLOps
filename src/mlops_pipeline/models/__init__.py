"""Model factory, evaluation, leakage checks and training."""

from .evaluation import ModelResult, compute_metrics, evaluate_model
from .factory import build_estimator, build_pipeline
from .training import TrainingRun, format_summary, run_training

__all__ = [
    "ModelResult", "TrainingRun", "build_estimator", "build_pipeline", "compute_metrics",
    "evaluate_model", "format_summary", "run_training",
]
