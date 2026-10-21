"""Cross-section validation of the configuration.

``schema.py`` guarantees structure and types; this module guarantees the
sections agree with each other (column roles vs. data contract, preprocessing
vs. features, paths) and reports settings that are intentionally still pending.
All functions return plain lists of messages so callers can aggregate them.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from .schema import AppConfig


def find_pending(raw: Any, prefix: str = "") -> list[str]:
    """Dotted paths of every ``null`` leaf in the raw YAML mapping."""
    pending: list[str] = []
    if isinstance(raw, dict):
        for key, value in raw.items():
            pending += find_pending(value, f"{prefix}.{key}" if prefix else str(key))
    elif raw is None:
        pending.append(prefix)
    return pending


def validate_semantics(
    cfg: AppConfig,
    *,
    project_root: Path | None = None,
    check_paths: bool = False,
) -> list[str]:
    """Return a list of problems (empty list = valid)."""
    problems: list[str] = []
    problems += _check_column_roles(cfg)
    problems += _check_contract_alignment(cfg)
    problems += _check_preprocessing(cfg)
    problems += _check_model(cfg)
    problems += check_data_validation_settings(cfg)
    problems += _check_paths(cfg, project_root, check_paths)
    if cfg.model_registry.enabled and not cfg.mlflow.enabled:
        problems.append("model_registry.enabled requires mlflow.enabled because registry versions must link to tracked runs")
    return problems


# ------------------------------------------------------------ column roles
def _check_column_roles(cfg: AppConfig) -> list[str]:
    problems: list[str] = []
    features = cfg.feature_columns
    excluded = cfg.excluded_columns

    for name, count in Counter(features).items():
        if count > 1:
            problems.append(f"columns.features: {name!r} is listed more than once")
    for name in set(cfg.numeric_features) & set(cfg.categorical_features):
        problems.append(f"columns.features: {name!r} is both numeric and categorical")
    if cfg.target in features:
        problems.append(f"columns.target {cfg.target!r} must not be a feature")
    if cfg.target in excluded:
        problems.append(f"columns.target {cfg.target!r} must not be in columns.excluded")
    for name in sorted(set(features) & set(excluded)):
        problems.append(f"{name!r} is both a feature and an excluded column")
    if not features:
        problems.append("columns.features: at least one feature is required")

    for role in ("identifier", "batch"):
        n = sum(1 for spec in cfg.columns.excluded.values() if spec.role == role)
        if n != 1:
            problems.append(f"columns.excluded: expected exactly one {role!r} column, found {n}")
    return problems


# ------------------------------------------------------ contract alignment
def _check_contract_alignment(cfg: AppConfig) -> list[str]:
    problems: list[str] = []
    spec = cfg.data_contract.columns
    assigned = {cfg.target, *cfg.feature_columns, *cfg.excluded_columns}

    for name in sorted(assigned - set(spec)):
        problems.append(f"column {name!r} has a role in 'columns' but is missing from data_contract")
    for name in sorted(set(spec) - assigned):
        problems.append(
            f"data_contract column {name!r} has no role: add it to columns.features, "
            "columns.excluded or columns.target"
        )

    both = {"train", "incoming"}
    t = spec.get(cfg.target)
    if t is not None:
        if t.dtype != "float":
            problems.append(f"target {cfg.target!r} must have dtype float for regression, got {t.dtype!r}")
        if t.nullable:
            problems.append(f"target {cfg.target!r} must not be nullable")
        if set(t.available_in) != both:
            problems.append(f"target {cfg.target!r} must be available in train and incoming")

    for name in cfg.numeric_features:
        s = spec.get(name)
        if s and s.dtype not in ("integer", "float"):
            problems.append(f"numeric feature {name!r} has non-numeric contract dtype {s.dtype!r}")
    for name in cfg.categorical_features:
        s = spec.get(name)
        if s and s.dtype != "categorical":
            problems.append(f"categorical feature {name!r} must have contract dtype 'categorical'")
    for name in cfg.feature_columns:
        s = spec.get(name)
        if s and set(s.available_in) != both:
            problems.append(f"feature {name!r} must be available in both train and incoming")

    try:
        ident = spec.get(cfg.identifier_column)
        if ident and set(ident.available_in) != both:
            problems.append(f"identifier {cfg.identifier_column!r} must be available in train and incoming")
        batch = spec.get(cfg.batch_column)
        if batch is not None:
            if batch.available_in != ["incoming"]:
                problems.append(f"batch column {cfg.batch_column!r} must be available in incoming only")
            if batch.dtype != "integer" or not batch.allowed_values:
                problems.append(
                    f"batch column {cfg.batch_column!r} needs dtype integer and allowed_values "
                    "(the expected batch numbers)"
                )
    except ValueError:
        pass  # role-count problem already reported by _check_column_roles
    return problems


# ----------------------------------------------------------- preprocessing
def _check_preprocessing(cfg: AppConfig) -> list[str]:
    problems: list[str] = []
    encoders = cfg.preprocessing.categorical.encoders
    cats = set(cfg.categorical_features)

    for name in sorted(cats - set(encoders)):
        problems.append(f"preprocessing.categorical.encoders: no encoder for feature {name!r}")
    for name in sorted(set(encoders) - cats):
        problems.append(f"preprocessing.categorical.encoders: {name!r} is not a categorical feature")

    spec = cfg.data_contract.columns
    for name, enc in encoders.items():
        col = spec.get(name)
        if enc.type == "ordinal" and col and col.allowed_values is not None:
            if set(enc.order or []) != set(col.allowed_values) or len(enc.order or []) != len(
                col.allowed_values
            ):
                problems.append(
                    f"encoder order for {name!r} must list exactly the contract's allowed_values"
                )
    return problems


# ------------------------------------------------------------------ model
def _check_model(cfg: AppConfig) -> list[str]:
    problems: list[str] = []
    ev = cfg.model.evaluation
    if ev.primary_metric in ev.secondary_metrics:
        problems.append("model.evaluation: primary_metric must not repeat in secondary_metrics")
    if len(set(ev.secondary_metrics)) != len(ev.secondary_metrics):
        problems.append("model.evaluation.secondary_metrics contains duplicates")
    if cfg.model.baseline.name == cfg.model.candidate.name:
        problems.append("model: baseline and candidate must have different names")
    for role in ("baseline", "candidate"):
        if "random_state" in cfg.model_spec(role).hyperparameters:
            problems.append(
                f"model.{role}.hyperparameters: do not set random_state; "
                "the seed comes from reproducibility.random_seed"
            )
    return problems


# ------------------------------------------------------------------- paths
def _check_paths(cfg: AppConfig, project_root: Path | None, check_files: bool) -> list[str]:
    problems: list[str] = []
    for key, value in cfg.paths.model_dump().items():
        if PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute():
            problems.append(f"paths.{key}: must be relative to the repository root, got {value!r}")
        elif ".." in PurePosixPath(value.replace("\\", "/")).parts:
            problems.append(f"paths.{key}: must not contain '..', got {value!r}")

    if check_files and project_root is not None:
        for key in ("train_data", "incoming_data"):
            path = project_root / getattr(cfg.paths, key)
            if not path.is_file():
                problems.append(f"paths.{key}: file not found at {path}")
    return problems


# ------------------------------------------------------- data validation
def check_data_validation_settings(cfg: AppConfig) -> list[str]:
    """Consistency of the ``validation`` section with the rest of the config.

    Run at config load and again when a ``DataValidator`` is built, so missing,
    malformed or contradictory settings can never reach a running gate.
    """
    v = cfg.validation
    spec = cfg.data_contract.columns
    problems: list[str] = []

    def role_columns(role: str) -> list[str]:
        try:
            return cfg.columns_for_role(role)
        except ValueError:  # role-count problem is reported by _check_column_roles
            return []

    # -- check toggles --------------------------------------------------
    c = v.checks
    dependants = [n for n in ("value_ranges", "missing_values", "categorical_values", "target",
                              "duplicates", "identifiers") if getattr(c, n)]
    if not c.schema_ and dependants:
        problems.append(
            f"validation.checks: 'schema' is disabled but {dependants} are enabled; those checks "
            "need the schema check to establish that columns exist"
        )

    # -- profiles --------------------------------------------------------
    used_columns: set[str] = set()
    for name, p in v.profiles.items():
        where = f"validation.profiles.{name}"
        for label, roles in (("required_roles", p.required_roles), ("optional_roles", p.optional_roles)):
            if len(set(roles)) != len(roles):
                problems.append(f"{where}.{label}: contains duplicates")
        both = set(p.required_roles) & set(p.optional_roles)
        if both:
            problems.append(f"{where}: roles {sorted(both)} are both required and optional")
        if "features" not in p.required_roles:
            problems.append(f"{where}.required_roles: must include 'features'")
        if name == "train" and "target" not in p.required_roles:
            problems.append(f"{where}.required_roles: the training profile must require the 'target'")
        if p.identifier_disjoint_from_training:
            if name == "train":
                problems.append(f"{where}: identifier_disjoint_from_training makes no sense for the training data itself")
            if "identifier" not in {*p.required_roles, *p.optional_roles}:
                problems.append(f"{where}: identifier_disjoint_from_training needs the 'identifier' role")
        for role in {*p.required_roles, *p.optional_roles}:
            for col in role_columns(role):
                col_spec = spec.get(col)
                if col_spec is None:
                    continue  # reported by the contract-alignment check
                used_columns.add(col)
                if p.contract_dataset not in col_spec.available_in:
                    problems.append(
                        f"{where}: role {role!r} column {col!r} is not part of the "
                        f"{p.contract_dataset!r} data contract"
                    )

    # -- numeric ranges ----------------------------------------------------
    for col, rng in v.numeric_ranges.items():
        s = spec.get(col)
        if s is None:
            problems.append(f"validation.numeric_ranges: {col!r} is not a data_contract column")
            continue
        if s.dtype not in ("integer", "float"):
            problems.append(f"validation.numeric_ranges: {col!r} is not numeric (dtype {s.dtype!r})")
        if s.allowed_values is not None:
            problems.append(f"validation.numeric_ranges: {col!r} already has allowed_values in the contract")
        if rng.min is not None and s.min is not None and rng.min < s.min:
            problems.append(f"validation.numeric_ranges.{col}: min {rng.min} is wider than the contract min {s.min}")
        if rng.max is not None and s.max is not None and rng.max > s.max:
            problems.append(f"validation.numeric_ranges.{col}: max {rng.max} is wider than the contract max {s.max}")
    if c.value_ranges:
        for col in sorted(used_columns):
            s = spec[col]
            if s.dtype not in ("integer", "float") or s.allowed_values is not None:
                continue
            rng = v.numeric_ranges.get(col)
            lows = [x for x in (s.min, rng.min if rng else None) if x is not None]
            highs = [x for x in (s.max, rng.max if rng else None) if x is not None]
            if not lows or not highs:
                problems.append(
                    f"validation.numeric_ranges: numeric column {col!r} has no complete [min, max] "
                    "range (data_contract and validation.numeric_ranges together must give both bounds)"
                )

    # -- missing-value limits ---------------------------------------------
    mv = v.missing_values
    per_col = mv.max_fraction_per_column
    for col, frac in per_col.items():
        s = spec.get(col)
        if s is None:
            problems.append(f"validation.missing_values.max_fraction_per_column: {col!r} is not a data_contract column")
        elif not s.nullable and frac > 0:
            problems.append(
                f"validation.missing_values.max_fraction_per_column.{col}: {frac} contradicts "
                f"data_contract ({col!r} is not nullable)"
            )
    if c.missing_values:
        for col in sorted(used_columns - {cfg.target}):
            if spec[col].nullable and col not in per_col:
                problems.append(
                    f"validation.missing_values.max_fraction_per_column: no limit for nullable column {col!r}"
                )
        if mv.max_overall_fraction == 0 and any(f > 0 for f in per_col.values()):
            problems.append(
                "validation.missing_values: max_overall_fraction is 0 but per-column limits above 0 "
                "allow missing values"
            )

    # -- identifiers ---------------------------------------------------------
    pattern = v.identifiers.pattern
    if pattern is not None:
        try:
            re.compile(pattern)
        except re.error as exc:
            problems.append(f"validation.identifiers.pattern: invalid regular expression ({exc})")
        ident = spec.get(cfg.identifier_column) if role_columns("identifier") else None
        if ident is not None and ident.dtype != "string":
            problems.append("validation.identifiers.pattern: the identifier column must have dtype string")
    return problems
