"""Phase 2 training run: split, fit baseline + candidate, evaluate, check, save.

No orchestration, tracking or registry here; later phases wrap this logic.
"""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from ..config import Settings
from ..data import (DataSplit, DataValidationError, DataValidator, load_training_data,
                    split_train_validation)
from .evaluation import (HIGHER_IS_BETTER, ModelResult, best_by_primary_metric,
                         evaluate_model, is_better)
from .leakage import LeakageCheck, check_leakage

ROLES = ("baseline", "candidate")


@dataclass
class TrainingRun:
    settings: Settings
    split: DataSplit
    results: dict[str, ModelResult]
    leakage: list[LeakageCheck]
    report: dict
    saved_paths: dict[str, Path]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _result_dict(r: ModelResult) -> dict:
    return {
        "name": r.spec.name,
        "algorithm": r.spec.algorithm,
        "hyperparameters": r.spec.hyperparameters,
        "n_features_after_preprocessing": r.n_features_after_preprocessing,
        "train_metrics": r.train_metrics,
        "cv_metrics_on_training_split": r.cv_metrics,
        "validation_metrics": r.validation_metrics,
    }


def _comparison(results: dict[str, ModelResult], cfg) -> dict:
    base, cand = results["baseline"], results["candidate"]
    primary = cfg.model.evaluation.primary_metric
    delta = {}
    for m in cfg.metric_names:
        b, c = base.validation_metrics[m], cand.validation_metrics[m]
        delta[m] = {"baseline": b, "candidate": c, "candidate_minus_baseline": c - b,
                    "candidate_relative_change": (c - b) / abs(b) if b else None,
                    "candidate_better": is_better(m, c, b)}
    # Same folds for both models (same seed), so fold-wise differences are paired.
    f_base = np.array(base.cv_metrics[primary]["per_fold"])
    f_cand = np.array(cand.cv_metrics[primary]["per_fold"])
    diff = f_cand - f_base
    candidate_wins = int((diff < 0).sum() if not HIGHER_IS_BETTER[primary] else (diff > 0).sum())
    cv_primary = {
        r: {k: results[r].cv_metrics[primary][k] for k in ("mean", "std")} for r in ROLES}
    return {
        "primary_metric": primary,
        "best_on_validation": best_by_primary_metric([base, cand], cfg).role,
        "validation": delta,
        "cv_primary_metric_mean_std": cv_primary,
        "cv_primary_metric_paired_difference": {
            "candidate_minus_baseline_mean": float(diff.mean()),
            "candidate_minus_baseline_std": float(diff.std(ddof=1)),
            "folds_where_candidate_better": candidate_wins,
            "n_folds": int(len(diff)),
        },
    }


def run_training(settings: Settings, *, save: bool = True) -> TrainingRun:
    cfg = settings.config
    df = load_training_data(settings)
    # Quality gate: must run before splitting, fitting, evaluation, or artifact writes.
    # DataValidationError carries the complete structured report for callers.
    validation_report = DataValidator.from_settings(settings).validate(df, "train")
    validation_report.raise_if_failed()
    split = split_train_validation(df, cfg)

    results = {role: evaluate_model(role, cfg.model_spec(role), split, cfg) for role in ROLES}
    leakage = check_leakage(cfg, split, df, list(results.values()))

    comparison = _comparison(results, cfg)
    selected_role = cfg.model.selected
    selection = {
        "selected_in_config": selected_role,
        "matches_best_on_validation": (
            None if selected_role is None else selected_role == comparison["best_on_validation"]),
    }

    saved: dict[str, Path] = {}
    if save:
        models_dir = settings.resolve(cfg.paths.models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)
        for role, r in results.items():
            path = models_dir / f"{r.spec.name}.joblib"
            joblib.dump(r.pipeline, path)
            saved[role] = path
        # Phase 6: this legacy training API is for candidate evaluation only.
        # It must never overwrite the production artifact; only a future explicit
        # promotion phase may do that after all gates pass.
    selection["selected_model_file"] = None
    selection["production_model_replaced"] = False

    report = {
        "settings": {
            "random_seed": cfg.random_seed,
            "test_size": cfg.model.split.test_size,
            "cv_folds": cfg.model.evaluation.cv_folds,
            "primary_metric": cfg.model.evaluation.primary_metric,
            "metrics": cfg.metric_names,
            "feature_columns": cfg.feature_columns,
            "excluded_columns": cfg.excluded_columns,
        },
        "data": {
            "training_file": cfg.paths.train_data,
            "training_file_sha256": _sha256(settings.train_path),
            "n_rows": len(df),
            **{f"n_{k}": v for k, v in split.sizes.items()},
        },
        "models": {role: _result_dict(r) for role, r in results.items()},
        "comparison": comparison,
        "selection": selection,
        "leakage_checks": [c.to_dict() for c in leakage],
        "environment": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }
    if save:
        out = settings.resolve(cfg.paths.evaluation_report)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=False) + "\n", encoding="utf-8")
        saved["report"] = out

    return TrainingRun(settings, split, results, leakage, report, saved)


def format_summary(run: TrainingRun) -> str:
    cfg = run.settings.config
    metrics = cfg.metric_names
    head = f"{'model':<30}{'split':<14}" + "".join(f"{m.upper():>10}" for m in metrics)
    lines = [
        f"Rows: train={run.split.sizes['train']}  validation={run.split.sizes['validation']}  "
        f"seed={cfg.random_seed}  primary metric={cfg.model.evaluation.primary_metric.upper()}",
        head, "-" * len(head),
    ]
    for role, r in run.results.items():
        rows = [("train", r.train_metrics), ("cv mean", {m: r.cv_metrics[m]["mean"] for m in metrics}),
                ("VALIDATION", r.validation_metrics)]
        for label, vals in rows:
            lines.append(f"{r.spec.name if label == 'train' else '':<30}{label:<14}"
                         + "".join(f"{vals[m]:>10.4f}" for m in metrics))
    cmp_ = run.report["comparison"]
    lines.append(f"\nBest on validation by {cmp_['primary_metric'].upper()}: {cmp_['best_on_validation']}")
    sel = run.report["selection"]["selected_in_config"]
    lines.append(f"Selected in config: {sel if sel else 'PENDING (model.selected is null)'}")
    failed = [c for c in run.leakage if not c.passed]
    lines.append(f"Leakage checks: {len(run.leakage) - len(failed)}/{len(run.leakage)} passed")
    lines += [f"  FAILED: {c.name}: {c.detail}" for c in failed]
    return "\n".join(lines)
