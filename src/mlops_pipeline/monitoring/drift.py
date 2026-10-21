"""Evidence-based data and performance monitoring for supplied incoming batches.

The monitor is intentionally separate from operational Prometheus/Grafana metrics.
It reads the supplied incoming CSV without modifying it, compares features with the
historical training reference, and evaluates the production model against the
provided target values.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from ..config import Settings
from ..data.splitting import split_train_validation
from ..models.evaluation import compute_metrics

EPSILON = 1e-6


@dataclass(frozen=True)
class MonitoringResult:
    report: dict[str, Any]
    output_paths: dict[str, Path]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _psi_from_probs(reference: np.ndarray, current: np.ndarray) -> float:
    ref = np.clip(np.asarray(reference, dtype=float), EPSILON, None)
    cur = np.clip(np.asarray(current, dtype=float), EPSILON, None)
    ref /= ref.sum()
    cur /= cur.sum()
    return float(np.sum((ref - cur) * np.log(ref / cur)))


def _numeric_bins(series: pd.Series) -> np.ndarray:
    values = series.dropna().to_numpy()
    if not len(values):
        return np.array([-np.inf, np.inf])
    quantiles = np.unique(np.quantile(values, np.linspace(0.0, 1.0, 11)))
    if len(quantiles) < 2:
        return np.array([-np.inf, np.inf])
    return np.r_[-np.inf, quantiles[1:-1], np.inf]


def _feature_spec(reference: pd.DataFrame, features: list[str], numeric: list[str]) -> dict[str, tuple[str, Any]]:
    spec: dict[str, tuple[str, Any]] = {}
    for column in features:
        if column in numeric:
            spec[column] = ("numeric", _numeric_bins(reference[column]))
        else:
            categories = sorted(set(reference[column].dropna().astype(str).unique()) | {"__MISSING__"})
            spec[column] = ("categorical", categories)
    return spec


def _feature_psi(reference: pd.DataFrame, current: pd.DataFrame, column: str, spec: dict[str, tuple[str, Any]]) -> float:
    kind, bins_or_categories = spec[column]
    if kind == "numeric":
        bins = bins_or_categories

        def distribution(frame: pd.DataFrame) -> np.ndarray:
            values = frame[column]
            missing = float(values.isna().mean())
            non_missing = values.dropna().to_numpy()
            counts, _ = np.histogram(non_missing, bins=bins)
            probs = counts / len(frame)
            return np.r_[probs, missing]

        return _psi_from_probs(distribution(reference), distribution(current))

    categories = bins_or_categories

    def distribution(frame: pd.DataFrame) -> np.ndarray:
        values = frame[column].astype(object).where(frame[column].notna(), "__MISSING__").astype(str)
        return values.value_counts().reindex(categories, fill_value=0).to_numpy(dtype=float) / len(frame)

    return _psi_from_probs(distribution(reference), distribution(current))


def _bootstrap_drift_thresholds(
    reference: pd.DataFrame,
    features: list[str],
    numeric_features: list[str],
    batch_size: int,
    seed: int,
    bootstrap_runs: int,
) -> tuple[dict[str, float], float, pd.DataFrame]:
    spec = _feature_spec(reference, features, numeric_features)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float]] = []
    for _ in range(bootstrap_runs):
        sample = reference.iloc[rng.integers(0, len(reference), size=batch_size)]
        rows.append({feature: _feature_psi(reference, sample, feature, spec) for feature in features})
    null = pd.DataFrame(rows)
    thresholds = {feature: float(null[feature].quantile(0.95)) for feature in features}
    null_share = null.gt(pd.Series(thresholds)).mean(axis=1)
    share_p95 = float(null_share.quantile(0.95))
    share_threshold = math.ceil(share_p95 * len(features)) / len(features)
    return thresholds, share_threshold, null


def _bootstrap_performance_threshold(
    model: Any,
    validation_X: pd.DataFrame,
    validation_y: pd.Series,
    baseline_mae: float,
    batch_size: int,
    seed: int,
    bootstrap_runs: int,
) -> tuple[float, np.ndarray]:
    # Predictions are fixed for the validation reference, so bootstrap the
    # absolute residuals directly instead of re-running the sklearn pipeline
    # thousands of times. This is mathematically identical for MAE and makes
    # the calibration fast enough for repeatable local monitoring.
    validation_predictions = model.predict(validation_X)
    residuals = np.abs(validation_y.to_numpy() - validation_predictions)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(residuals), size=(bootstrap_runs, batch_size))
    sample_mae = residuals[indices].mean(axis=1)
    arr = (sample_mae - baseline_mae) / baseline_mae
    return float(np.quantile(arr, 0.95)), arr


def monitor_incoming_batches(settings: Settings, *, output_dir: str | Path | None = None) -> MonitoringResult:
    """Run evidence-based feature and performance monitoring on all incoming batches."""
    cfg = settings.config
    reference_path = settings.train_path
    incoming_path = settings.incoming_path
    model_path = settings.resolve(cfg.paths.selected_model)
    out_dir = settings.resolve(output_dir or "reports/phase14_evidence")
    out_dir.mkdir(parents=True, exist_ok=True)

    reference = pd.read_csv(reference_path)
    incoming_before_hash = _sha256(incoming_path)
    incoming = pd.read_csv(incoming_path)
    incoming_after_read_hash = _sha256(incoming_path)
    if incoming_before_hash != incoming_after_read_hash:
        raise RuntimeError("incoming dataset changed while being monitored")
    if cfg.batch_column not in incoming.columns:
        raise ValueError(f"incoming dataset does not contain batch column {cfg.batch_column!r}")

    batch_sizes = incoming.groupby(cfg.batch_column, sort=True).size()
    if batch_sizes.empty:
        raise ValueError("incoming dataset contains no batches")
    if batch_sizes.nunique() != 1:
        raise ValueError(f"incoming batches must have equal sizes for calibrated monitoring: {batch_sizes.to_dict()}")
    batch_size = int(batch_sizes.iloc[0])
    drift_cfg = cfg.monitoring.data_drift
    performance_cfg = cfg.monitoring.performance_drift
    if drift_cfg.bootstrap_batch_size > len(reference):
        raise ValueError("configured data-drift bootstrap batch size exceeds training-reference size")
    split = split_train_validation(reference, cfg)
    if performance_cfg.bootstrap_batch_size > len(split.y_valid):
        raise ValueError("configured performance bootstrap batch size exceeds validation size")

    model = joblib.load(model_path)
    validation_predictions = model.predict(split.X_valid)
    baseline_metrics = compute_metrics(split.y_valid, validation_predictions, ["mae", "rmse", "r2"])

    feature_thresholds, share_threshold, null_drift = _bootstrap_drift_thresholds(
        reference, cfg.feature_columns, cfg.numeric_features,
        drift_cfg.bootstrap_batch_size, cfg.random_seed, drift_cfg.bootstrap_runs
    )
    performance_threshold, null_performance = _bootstrap_performance_threshold(
        model, split.X_valid, split.y_valid, baseline_metrics["mae"],
        performance_cfg.bootstrap_batch_size, cfg.random_seed + 1, performance_cfg.bootstrap_runs
    )
    spec = _feature_spec(reference, cfg.feature_columns, cfg.numeric_features)

    feature_rows: list[dict[str, Any]] = []
    batch_rows: list[dict[str, Any]] = []
    for batch_id, batch in incoming.groupby(cfg.batch_column, sort=True):
        psis = {feature: _feature_psi(reference, batch, feature, spec) for feature in cfg.feature_columns}
        drifted = [feature for feature, psi in psis.items() if psi > feature_thresholds[feature]]
        drift_share = len(drifted) / len(cfg.feature_columns)

        X = batch[cfg.feature_columns]
        y = batch[cfg.target]
        predictions = model.predict(X)
        metrics = compute_metrics(y, predictions, ["mae", "rmse", "r2"])
        relative_mae_degradation = (metrics["mae"] - baseline_metrics["mae"]) / baseline_metrics["mae"]

        for feature in cfg.feature_columns:
            feature_rows.append({
                "batch": int(batch_id),
                "feature": feature,
                "psi": psis[feature],
                "null_95_threshold": feature_thresholds[feature],
                "drift": psis[feature] > feature_thresholds[feature],
            })

        batch_rows.append({
            "batch": int(batch_id),
            "rows": len(batch),
            "drifted_feature_count": len(drifted),
            "drifted_feature_share": drift_share,
            "data_drift_trigger": drift_share >= share_threshold,
            "drifted_features": drifted,
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "r2": metrics["r2"],
            "relative_mae_degradation": relative_mae_degradation,
            "performance_drift_trigger": relative_mae_degradation > performance_threshold,
            "actual_price_mean": float(y.mean()),
            "actual_price_median": float(y.median()),
            "prediction_mean": float(np.mean(predictions)),
            "mean_prediction_bias": float(np.mean(predictions - y.to_numpy())),
        })

    # Preserve the incoming file's hash as a guardrail/evidence item.
    final_hash = _sha256(incoming_path)
    if final_hash != incoming_before_hash:
        raise RuntimeError("incoming dataset changed during monitoring")

    report = {
        "status": "PASS",
        "methodology": {
            "reference_dataset": str(reference_path.relative_to(settings.project_root)),
            "reference_rows": len(reference),
            "incoming_dataset": str(incoming_path.relative_to(settings.project_root)),
            "incoming_sha256_before": incoming_before_hash,
            "incoming_sha256_after": final_hash,
            "incoming_unchanged": incoming_before_hash == final_hash,
            "batch_size": batch_size,
            "feature_drift_method": "Population Stability Index (PSI)",
            "numeric_psi_bins": "10 quantile bins learned from the full training reference, plus a missingness bin",
            "categorical_psi_categories": "training-reference categories plus a missingness bin",
            "null_drift_bootstrap_runs": drift_cfg.bootstrap_runs,
            "drift_bootstrap_batch_size": drift_cfg.bootstrap_batch_size,
            "per_feature_threshold_rule": "95th percentile of the empirical null PSI distribution from the configured training bootstrap sample size",
            "batch_drift_share_threshold_rule": "95th percentile of null drifted-feature share, rounded up to the next whole-feature increment",
            "performance_model": str(model_path.relative_to(settings.project_root)),
            "performance_reference": "fixed seed-42 validation split from the supplied training dataset",
            "performance_metric": "MAE on actual target values; RMSE and R2 are also reported",
            "performance_null_bootstrap_runs": performance_cfg.bootstrap_runs,
            "performance_bootstrap_batch_size": performance_cfg.bootstrap_batch_size,
            "performance_threshold_rule": "95th percentile of relative MAE degradation from the configured validation bootstrap sample size",
            "baseline_validation_metrics": baseline_metrics,
            "per_feature_thresholds": feature_thresholds,
            "data_drift_share_threshold": share_threshold,
            "performance_relative_mae_degradation_threshold": performance_threshold,
            "production_model_replacement": "not performed; drift/performance alerts do not bypass the Model Gate",
        },
        "batches": batch_rows,
        "feature_results": feature_rows,
    }

    json_path = out_dir / "drift_results.json"
    batch_csv = out_dir / "batch_summary.csv"
    feature_csv = out_dir / "feature_drift.csv"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(batch_rows).to_csv(batch_csv, index=False)
    pd.DataFrame(feature_rows).to_csv(feature_csv, index=False)

    _write_markdown_report(out_dir / "phase14_monitoring_report.md", report)
    _write_charts(out_dir, batch_rows, feature_rows, cfg.feature_columns)

    return MonitoringResult(
        report=report,
        output_paths={
            "json": json_path,
            "batch_csv": batch_csv,
            "feature_csv": feature_csv,
            "markdown": out_dir / "phase14_monitoring_report.md",
        },
    )


def _write_markdown_report(path: Path, report: dict[str, Any]) -> None:
    m = report["methodology"]
    lines = [
        "# Phase 14 — ML/Data Monitoring Results",
        "",
        "## Methodology",
        f"- Reference: `{m['reference_dataset']}` ({m['reference_rows']} rows).",
        f"- Incoming: `{m['incoming_dataset']}` ({m['batch_size']} rows per batch); SHA-256 unchanged: **{m['incoming_unchanged']}**.",
        f"- Feature drift: {m['feature_drift_method']} with {m['numeric_psi_bins']}; categorical features use {m['categorical_psi_categories']}.",
        f"- Per-feature PSI threshold: empirical 95th percentile from {m['null_drift_bootstrap_runs']} training bootstrap samples of {m['drift_bootstrap_batch_size']} rows.",
        f"- Batch data-drift threshold: {m['data_drift_share_threshold']:.3f} of features ({int(round(m['data_drift_share_threshold'] * len(report['feature_results']) / 3))} of 9), calibrated from the null drift-share distribution.",
        f"- Performance reference: fixed seed-42 validation split; baseline MAE = {m['baseline_validation_metrics']['mae']:.4f} lakhs.",
        f"- Performance threshold: {m['performance_relative_mae_degradation_threshold']:.2%} relative MAE degradation, calibrated from {m['performance_null_bootstrap_runs']} validation bootstrap samples of {m['performance_bootstrap_batch_size']} rows.",
        "",
        "## Batch results",
        "",
        "| Batch | Data drift | Performance drift | Drifted features | MAE | RMSE | R² | Relative MAE degradation |",
        "|---:|:---:|:---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["batches"]:
        lines.append(
            f"| {row['batch']} | {'YES' if row['data_drift_trigger'] else 'NO'} | "
            f"{'YES' if row['performance_drift_trigger'] else 'NO'} | {row['drifted_feature_count']} | "
            f"{row['mae']:.4f} | {row['rmse']:.4f} | {row['r2']:.4f} | {row['relative_mae_degradation']:.2%} |"
        )
    lines += ["", "## Interpretation"]
    for row in report["batches"]:
        status = []
        if row["data_drift_trigger"]:
            status.append("feature distribution drift")
        if row["performance_drift_trigger"]:
            status.append("model performance degradation")
        label = ", ".join(status) if status else "no monitoring alert"
        lines.append(
            f"- **Batch {row['batch']}**: {label}. Drifted features: "
            f"{', '.join(row['drifted_features']) if row['drifted_features'] else 'none'}. "
            f"Actual mean price = {row['actual_price_mean']:.2f} lakhs; mean prediction bias = {row['mean_prediction_bias']:.2f} lakhs."
        )
    lines += [
        "",
        "## Response policy",
        "- A drift alert is a monitoring signal, not permission to replace production.",
        "- Investigate the affected batch, confirm data quality and business change, and consider retraining when sustained or material.",
        "- Any new model must still pass the existing Data Gate and Model Gate before registry promotion.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_charts(out_dir: Path, batch_rows: list[dict[str, Any]], feature_rows: list[dict[str, Any]], features: list[str]) -> None:
    import matplotlib.pyplot as plt

    batches = [str(r["batch"]) for r in batch_rows]
    drift_share = [r["drifted_feature_share"] for r in batch_rows]
    perf_deg = [r["relative_mae_degradation"] for r in batch_rows]

    fig = plt.figure(figsize=(8, 4.5))
    ax = fig.add_subplot(111)
    ax.bar(batches, drift_share)
    ax.set_title("Feature Drift Share by Incoming Batch")
    ax.set_xlabel("Incoming group")
    ax.set_ylabel("Share of features")
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(out_dir / "feature_drift_share.png", dpi=160)
    plt.close(fig)

    fig = plt.figure(figsize=(8, 4.5))
    ax = fig.add_subplot(111)
    ax.bar(batches, perf_deg)
    ax.set_title("Relative MAE Degradation by Incoming Batch")
    ax.set_xlabel("Incoming group")
    ax.set_ylabel("Relative degradation")
    ax.axhline(0, linewidth=1)
    fig.tight_layout()
    fig.savefig(out_dir / "performance_drift.png", dpi=160)
    plt.close(fig)

    table = pd.DataFrame(feature_rows).pivot(index="feature", columns="batch", values="psi").reindex(features)
    fig = plt.figure(figsize=(8.5, 5.2))
    ax = fig.add_subplot(111)
    image = ax.imshow(table.to_numpy(), aspect="auto")
    ax.set_title("PSI by Feature and Incoming Batch")
    ax.set_xticks(range(len(table.columns)), [str(c) for c in table.columns])
    ax.set_yticks(range(len(table.index)), table.index)
    fig.colorbar(image, ax=ax, label="PSI")
    fig.tight_layout()
    fig.savefig(out_dir / "psi_heatmap.png", dpi=160)
    plt.close(fig)
