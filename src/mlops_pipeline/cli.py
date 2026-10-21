"""Command line entry points: ``python -m mlops_pipeline <command>``."""

from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from .config import ConfigError, load_config
from .data import DataContract, DataValidationError


def _cmd_validate_config(args: argparse.Namespace) -> int:
    try:
        settings = load_config(args.config, check_paths=not args.skip_path_check)
    except ConfigError as exc:
        print(exc, file=sys.stderr)
        return 1
    cfg = settings.config
    print(f"Configuration VALID: {settings.config_path}")
    print(f"  target            : {cfg.target}")
    print(f"  features ({len(cfg.feature_columns):>2})     : {', '.join(cfg.feature_columns)}")
    print(f"  excluded          : {', '.join(cfg.excluded_columns)}")
    print(f"  contract columns  : {len(cfg.data_contract.columns)}")
    print(f"  pending settings  : {len(settings.pending)} (intentionally null; decided in later phases)")
    for item in settings.pending:
        print(f"    - {item}")
    return 0


def _cmd_check_contract(args: argparse.Namespace) -> int:
    try:
        settings = load_config(args.config, check_paths=True)
    except ConfigError as exc:
        print(exc, file=sys.stderr)
        return 1
    contract = DataContract.from_config(settings.config)
    failed = False
    for dataset, path in (("train", settings.train_path), ("incoming", settings.incoming_path)):
        report = contract.check(pd.read_csv(path), dataset)
        print(report.to_text())
        failed |= not report.ok
    return 1 if failed else 0


def _cmd_train(args: argparse.Namespace) -> int:
    from .models import format_summary, run_training  # lazy: keeps config commands light

    try:
        settings = load_config(args.config, check_paths=True)
    except ConfigError as exc:
        print(exc, file=sys.stderr)
        return 1
    try:
        run = run_training(settings)
    except DataValidationError as exc:
        print(exc.report.to_text(), file=sys.stderr)
        return 1
    print(format_summary(run))
    for label, path in run.saved_paths.items():
        print(f"saved {label:<10}: {path.relative_to(settings.project_root)}")
    return 1 if any(not c.passed for c in run.leakage) else 0


def _cmd_orchestrate(args: argparse.Namespace) -> int:
    from .pipeline.prefect_flow import PREFECT_AVAILABLE, house_price_flow

    if not PREFECT_AVAILABLE:
        print(
            "WARNING: Prefect is not installed in this environment; using the local "
            "stage-compatible fallback. Install requirements.txt to enable Prefect task "
            "tracking and Prefect UI visibility.",
            file=sys.stderr,
        )
    try:
        summary = house_price_flow(config_path=args.config, data_path=args.data_path)
    except Exception as exc:
        print(f"Prefect flow FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print("Prefect flow SUCCEEDED")
    print(pd.Series(summary).to_json(indent=2))
    return 1 if summary.get("leakage_failures") or summary.get("status") == "model_gate_failed" else 0



def _cmd_rollback_production(args: argparse.Namespace) -> int:
    from .pipeline.prefect_flow import PREFECT_AVAILABLE, rollback_production_flow
    if not PREFECT_AVAILABLE:
        print("ERROR: Prefect is required for controlled production rollback", file=sys.stderr)
        return 1
    try:
        result = rollback_production_flow(
            config_path=args.config, target_version=str(args.target_version),
            expected_current_version=str(args.expected_current_version) if args.expected_current_version else None,
            reason=args.reason,
        )
    except Exception as exc:
        print(f"Production rollback FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _cmd_monitor_batches(args: argparse.Namespace) -> int:
    try:
        settings = load_config(args.config, check_paths=True)
    except ConfigError as exc:
        print(exc, file=sys.stderr)
        return 1
    from .monitoring.drift import monitor_incoming_batches
    try:
        result = monitor_incoming_batches(settings, output_dir=args.output_dir)
    except Exception as exc:
        print(f"Incoming-batch monitoring FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result.report, indent=2, sort_keys=True))
    return 0


def _cmd_verify_lineage(args: argparse.Namespace) -> int:
    try:
        settings = load_config(args.config, check_paths=True)
    except ConfigError as exc:
        print(exc, file=sys.stderr)
        return 1
    from .monitoring.lineage import verify_production_lineage
    report = verify_production_lineage(settings)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


def _cmd_reproducibility_check(args: argparse.Namespace) -> int:
    try:
        settings = load_config(args.config, check_paths=True)
    except ConfigError as exc:
        print(exc, file=sys.stderr)
        return 1
    from .monitoring.reproducibility import check_reproducibility
    try:
        report = check_reproducibility(settings)
    except Exception as exc:
        print(f"Reproducibility check FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mlops_pipeline", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("validate-config", help="load and validate config/config.yaml")
    p.add_argument("--config", help="path to config file (default: config/config.yaml)")
    p.add_argument("--skip-path-check", action="store_true", help="do not require data files to exist")
    p.set_defaults(func=_cmd_validate_config)

    p = sub.add_parser("check-contract", help="check raw CSVs against the data contract")
    p.add_argument("--config", help="path to config file (default: config/config.yaml)")
    p.set_defaults(func=_cmd_check_contract)

    p = sub.add_parser("train", help="train + evaluate baseline and candidate on the training file")
    p.add_argument("--config", help="path to config file (default: config/config.yaml)")
    p.set_defaults(func=_cmd_train)

    p = sub.add_parser("orchestrate", help="run the Prefect-orchestrated ML pipeline")
    p.add_argument("--config", help="path to config file (default: config/config.yaml)")
    p.add_argument("--data-path", help="optional CSV override for controlled runs/tests")
    p.set_defaults(func=_cmd_orchestrate)

    p = sub.add_parser("rollback-production", help="rollback production registry alias through the Prefect pipeline")
    p.add_argument("target_version", help="known-good MLflow registered-model version to restore")
    p.add_argument("--expected-current-version", help="safety check: current production version that must be replaced")
    p.add_argument("--reason", default="rollback to previous known-good production version")
    p.add_argument("--config", help="path to config file (default: config/config.yaml)")
    p.set_defaults(func=_cmd_rollback_production)

    p = sub.add_parser("monitor-batches", help="measure feature drift and performance drift on supplied incoming batches")
    p.add_argument("--config", help="path to config file (default: config/config.yaml)")
    p.add_argument("--output-dir", help="output directory (default: reports/phase14_evidence)")
    p.set_defaults(func=_cmd_monitor_batches)

    p = sub.add_parser("verify-lineage", help="audit production model -> MLflow version/run -> lineage artifacts")
    p.add_argument("--config", help="path to config file (default: config/config.yaml)")
    p.set_defaults(func=_cmd_verify_lineage)

    p = sub.add_parser("reproducibility-check", help="train twice with identical inputs and compare metrics")
    p.add_argument("--config", help="path to config file (default: config/config.yaml)")
    p.set_defaults(func=_cmd_reproducibility_check)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)
