"""Independent model-quality gate; this module never promotes or writes models."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ..config.loader import Settings
from ..config.schema import AppConfig
from .evaluation import HIGHER_IS_BETTER


def _sha256(path: Path) -> str:
    """Return the SHA-256 hash of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _better_or_equal(metric: str, value: float, threshold: float) -> bool:
    return (
        value >= threshold
        if HIGHER_IS_BETTER[metric]
        else value <= threshold
    )


def evaluate_model_gate(
    candidate_metrics: dict[str, float],
    cfg: AppConfig,
    production_record: dict[str, Any] | None,
) -> dict[str, Any]:
    """Evaluate absolute quality and production comparison independently."""
    primary = cfg.model.evaluation.primary_metric
    if primary not in candidate_metrics:
        raise ValueError(
            f"Candidate metrics do not contain primary metric {primary!r}"
        )

    candidate_value = float(candidate_metrics[primary])
    threshold = float(cfg.model_gate.quality_threshold)
    higher_is_better = HIGHER_IS_BETTER[primary]
    absolute_pass = _better_or_equal(primary, candidate_value, threshold)

    absolute = {
        "status": "PASS" if absolute_pass else "FAIL",
        "metric": primary,
        "observed": candidate_value,
        "threshold": threshold,
        "direction": (
            "higher_is_better" if higher_is_better else "lower_is_better"
        ),
        "reason": (
            f"{primary.upper()} {candidate_value:.6f} satisfies absolute "
            f"threshold {threshold:.6f}"
            if absolute_pass
            else f"{primary.upper()} {candidate_value:.6f} does not satisfy "
            f"absolute threshold {threshold:.6f}"
        ),
    }

    comparison_cfg = cfg.model_gate.production_comparison

    if not comparison_cfg.enabled:
        comparison = {
            "status": "SKIPPED",
            "enabled": False,
            "production_metrics": None,
            "min_improvement": comparison_cfg.min_improvement,
            "reason": "Production comparison is disabled by config.yaml.",
        }
    elif production_record is None:
        comparison = {
            "status": "SKIPPED",
            "enabled": True,
            "production_metrics": None,
            "min_improvement": comparison_cfg.min_improvement,
            "reason": (
                "First-model case: no existing production model and metrics "
                "were found; absolute quality check is required."
            ),
        }
    else:
        production_metrics = production_record.get("metrics", {})
        if primary not in production_metrics:
            raise ValueError(
                f"Production record does not contain primary metric {primary!r}"
            )

        production_value = float(production_metrics[primary])
        min_improvement = float(comparison_cfg.min_improvement)

        # Even with zero minimum improvement, require strict improvement.
        comparison_pass = (
            candidate_value > production_value + min_improvement
            if higher_is_better
            else candidate_value < production_value - min_improvement
        )

        comparison = {
            "status": "PASS" if comparison_pass else "FAIL",
            "enabled": True,
            "metric": primary,
            "candidate_value": candidate_value,
            "production_value": production_value,
            "production_metrics": production_metrics,
            "min_improvement": min_improvement,
            "production_model": production_record.get("model_name", "unknown"),
            "production_artifact": production_record.get("artifact_path"),
            "reason": (
                f"Candidate improves {primary.upper()} over production by "
                f"{abs(production_value - candidate_value):.6f}."
                if comparison_pass
                else (
                    f"Candidate does not improve {primary.upper()} over "
                    f"production by the required minimum {min_improvement:.6f}; "
                    f"candidate={candidate_value:.6f}, "
                    f"production={production_value:.6f}."
                )
            ),
        }

    reasons = []
    if not absolute_pass:
        reasons.append(absolute["reason"])
    if comparison["status"] == "FAIL":
        reasons.append(comparison["reason"])

    passed = absolute_pass and comparison["status"] != "FAIL"

    return {
        "status": "PASS" if passed else "FAIL",
        "passed": passed,
        "primary_metric": primary,
        "candidate_metrics": {
            key: float(value) for key, value in candidate_metrics.items()
        },
        "absolute_quality_check": absolute,
        "production_comparison": comparison,
        "failure_reasons": reasons,
        "candidate_disposition": (
            "eligible_for_future_promotion" if passed else "rejected"
        ),
        "promotion_performed": False,
        "production_model_replaced": False,
        "production_model_preserved": True,
    }


def load_production_record(
    settings: Settings,
) -> dict[str, Any] | None:
    """Load production metric metadata and verify its model artifact.

    If a production artifact exists without trustworthy metadata, fail closed.
    """
    metadata_path = settings.resolve(
        settings.config.model_gate.production_metadata_path
    )
    production_path = settings.resolve(settings.config.paths.selected_model)

    if not metadata_path.is_file():
        if production_path.is_file():
            raise RuntimeError(
                f"Production model exists at {production_path} but its "
                f"metadata is missing at {metadata_path}; refusing to treat "
                "this as a first-model run."
            )
        return None

    try:
        record = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Cannot read production model metadata {metadata_path}: {exc}"
        ) from exc

    if record.get("production_mirror_status") == "registry_authoritative":
        if not record.get("registered_model_name") or not record.get(
            "model_version"
        ):
            raise RuntimeError(
                "Registry-authoritative production metadata is missing "
                "registry identity"
            )
        return record

    if not production_path.is_file():
        raise RuntimeError(
            f"Production metadata exists but model artifact is missing: "
            f"{production_path}"
        )

    expected_hash = record.get("artifact_sha256")
    actual_hash = _sha256(production_path)

    if not expected_hash or expected_hash != actual_hash:
        raise RuntimeError(
            f"Production model artifact hash mismatch for {production_path}; "
            "refusing unsafe comparison."
        )

    record["artifact_path"] = str(
        production_path.relative_to(settings.project_root)
    )
    return record


def write_model_gate_report(
    settings: Settings,
    result: dict[str, Any],
) -> Path:
    """Write an inspectable gate decision without modifying production."""
    report_path = (
        settings.resolve(settings.config.paths.reports_dir) / "model_gate.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report_path
