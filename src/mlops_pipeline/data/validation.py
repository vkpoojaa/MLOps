"""Data validation layer (the checks behind the Task 4 data gate).

What this is
------------
A reusable, side-effect free validator. ``DataValidator.validate(df, profile)``
inspects a DataFrame and returns a :class:`ValidationReport`: one
:class:`CheckResult` per individual check (name, status, relevant column,
observed value, expected value / threshold, failure reason) plus an overall
pass/fail. It never modifies the DataFrame, never reads files and never decides
what the pipeline does with a failure - enforcing the gate (halting, keeping the
previous model) is a separate step that consumes the report.

What this is not
----------------
* Not preprocessing: nothing is imputed, clipped, encoded or dropped.
* Not ``DataContract``: the contract (``contract.py``) stays a structural
  conformance report used by the loader. This layer *reads the same YAML column
  specs* (dtype, nullability, allowed values, uniqueness, brief-documented
  min/max) so they are defined once, and adds the rules the contract does not
  hold (plausibility ranges, missing-value limits, target and duplicate rules,
  per-profile column requirements).
* Not drift detection: ranges are corruption / plausibility guards.

Profiles (``validation.profiles`` in the YAML)
----------------------------------------------
``train``, ``incoming`` and ``inference`` each declare which column *roles*
(features, target, identifier, batch) are required or optional. The inference
profile only requires the features, so a scoring request without a price or
identifier is valid, while the same file under ``train`` is not.

Everything configurable (thresholds, ranges, categories, rules) comes from the
YAML. Misconfiguration raises ``ConfigError`` (at load, and again when the
validator is built). Status values: ``pass``, ``fail``, ``skipped``; ``skipped``
is never counted as a failure but is always listed, with the reason.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
from pandas.api import types as pdt

from ..config.loader import ConfigError, Settings
from ..config.schema import AppConfig, ColumnSpec, DatasetProfile
from ..config.validation import check_data_validation_settings
from .contract import DataContract, _dtype_problem

_EXAMPLES = 5  # max example values quoted in a result


class DataValidationError(Exception):
    """Raised by ``ValidationReport.raise_if_failed``; carries the report."""

    def __init__(self, report: "ValidationReport"):
        self.report = report
        super().__init__(report.to_text())


class CheckStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: CheckStatus
    column: str | None = None
    observed: Any = None
    expected: Any = None
    reason: str | None = None  # why it failed / was skipped; None when it passed

    @property
    def passed(self) -> bool:
        return self.status is CheckStatus.PASS

    @property
    def failed(self) -> bool:
        return self.status is CheckStatus.FAIL

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "column": self.column,
            "observed": self.observed,
            "expected": self.expected,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ValidationReport:
    profile: str
    n_rows: int
    n_columns: int
    results: tuple[CheckResult, ...]

    @property
    def passed(self) -> bool:
        """Overall result: True only if no check failed (and at least one ran)."""
        return bool(self.results) and not self.failures

    @property
    def failures(self) -> tuple[CheckResult, ...]:
        return tuple(r for r in self.results if r.failed)

    @property
    def counts(self) -> dict[str, int]:
        return {s.value: sum(1 for r in self.results if r.status is s) for s in CheckStatus}

    def get(self, name: str, column: str | None = None) -> CheckResult:
        """The single result with this check name (and column, if given)."""
        hits = [r for r in self.results if r.name == name and (column is None or r.column == column)]
        if len(hits) != 1:
            raise KeyError(f"expected exactly one result for check {name!r} column {column!r}, found {len(hits)}")
        return hits[0]

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "passed": self.passed,
            "n_rows": self.n_rows,
            "n_columns": self.n_columns,
            "counts": self.counts,
            "checks": [r.to_dict() for r in self.results],
        }

    def to_text(self) -> str:
        c = self.counts
        head = (f"Data validation - profile '{self.profile}' ({self.n_rows} rows, {self.n_columns} columns): "
                f"{'PASSED' if self.passed else 'FAILED'} "
                f"[{c['pass']} passed, {c['fail']} failed, {c['skipped']} skipped]")
        lines = [head]
        for r in self.failures:
            where = f"[{r.column}] " if r.column else ""
            lines.append(f"  FAIL {r.name}: {where}{r.reason} (observed={r.observed!r}, expected={r.expected!r})")
        return "\n".join(lines)

    def raise_if_failed(self) -> None:
        if not self.passed:
            raise DataValidationError(self)


# ----------------------------------------------------------------- helpers
def _py(value: Any) -> Any:
    """JSON-friendly Python value (numpy scalars -> Python, NaN/inf -> None)."""
    if value is None:
        return None
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            value = value.item()
        except (ValueError, TypeError):
            pass
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _pct(fraction: float) -> str:
    return f"{fraction * 100:.2f}%"


def _examples(values: Iterable[Any]) -> list[str]:
    return sorted({str(v) for v in values})[:_EXAMPLES]


def _effective_range(spec: ColumnSpec, rng) -> tuple[float | None, float | None]:
    """Combine the contract's (brief-documented) bounds with the plausibility range."""
    lows = [x for x in (spec.min, getattr(rng, "min", None)) if x is not None]
    highs = [x for x in (spec.max, getattr(rng, "max", None)) if x is not None]
    return (max(lows) if lows else None, min(highs) if highs else None)


def _skip(name: str, reason: str, column: str | None = None) -> CheckResult:
    return CheckResult(name, CheckStatus.SKIPPED, column, reason=reason)


def _result(name: str, ok: bool, column: str | None, observed: Any, expected: Any, reason: str) -> CheckResult:
    return CheckResult(name, CheckStatus.PASS if ok else CheckStatus.FAIL, column,
                       _py(observed), _py(expected), None if ok else reason)


class _Context:
    """Everything the check groups need about one validation call (read-only)."""

    def __init__(self, df: pd.DataFrame, cfg: AppConfig, contract: DataContract, profile_name: str):
        self.df = df
        self.cfg = cfg
        self.v = cfg.validation
        self.profile_name = profile_name
        self.profile: DatasetProfile = cfg.validation.profiles[profile_name]
        self.n_rows = len(df)
        self.role_of: dict[str, str] = {}
        self.required: dict[str, str] = {}
        for role in (*self.profile.required_roles, *self.profile.optional_roles):
            for col in cfg.columns_for_role(role):
                self.role_of[col] = role
                if role in self.profile.required_roles:
                    self.required[col] = role
        specs = contract.columns_for(self.profile.contract_dataset)
        self.specs: dict[str, ColumnSpec] = {c: specs[c] for c in self.role_of if c in specs}
        labels = pd.Index(df.columns)
        label_set = set(labels)
        self.labels = label_set
        self.duplicate_labels: set[Any] = set(labels[labels.duplicated()])
        self.present: set[str] = {c for c in self.role_of if c in label_set and c not in self.duplicate_labels}
        self.dtype_problem: dict[str, str | None] = {
            c: _dtype_problem(df[c], self.specs[c]) for c in self.present if c in self.specs}

    def series(self, col: str) -> pd.Series:
        return self.df[col]

    def dtype_ok(self, col: str) -> bool:
        return col in self.dtype_problem and self.dtype_problem[col] is None

    def numeric_ok(self, col: str) -> bool:
        """Numeric (non-bool) dtype: enough to look for infinities and out-of-range values even
        when the stricter contract dtype rule (e.g. integer-valued) already failed."""
        s = self.df[col]
        return col in self.present and pdt.is_numeric_dtype(s) and not pdt.is_bool_dtype(s)

    def is_target(self, col: str) -> bool:
        return col == self.cfg.target


# ---------------------------------------------------------------- validator
class DataValidator:
    """Config-driven data validation. Build once, call ``validate`` per dataset."""

    def __init__(self, cfg: AppConfig):
        problems = check_data_validation_settings(cfg)
        if problems:
            raise ConfigError(problems)
        self._cfg = cfg
        self._contract = DataContract.from_config(cfg)

    @classmethod
    def from_config(cls, cfg: AppConfig) -> "DataValidator":
        return cls(cfg)

    @classmethod
    def from_settings(cls, settings: Settings) -> "DataValidator":
        return cls(settings.config)

    @property
    def profiles(self) -> list[str]:
        return list(self._cfg.validation.profiles)

    def validate(
        self,
        df: pd.DataFrame,
        profile: str,
        *,
        reference_ids: Iterable[Any] | None = None,
    ) -> ValidationReport:
        """Validate ``df`` against ``profile`` and return a structured report.

        ``reference_ids`` are the training identifiers; they are required when the
        profile asks for ``identifier_disjoint_from_training`` (missing = failed check,
        never a silent pass). ``df`` is never modified.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"expected a pandas DataFrame, got {type(df).__name__}")
        if profile not in self._cfg.validation.profiles:
            raise ValueError(f"unknown validation profile {profile!r}; configured: {self.profiles}")
        if isinstance(reference_ids, pd.DataFrame):
            raise TypeError("reference_ids must be a sequence/Series of identifiers, not a DataFrame")

        ctx = _Context(df, self._cfg, self._contract, profile)
        checks = ctx.v.checks
        groups: list[tuple[str, bool, Callable[[], list[CheckResult]]]] = [
            ("schema", checks.schema_, lambda: self._schema(ctx)),
            ("missing_values", checks.missing_values, lambda: self._missing(ctx)),
            ("value_ranges", checks.value_ranges, lambda: self._ranges(ctx)),
            ("categorical_values", checks.categorical_values, lambda: self._categories(ctx)),
            ("target", checks.target, lambda: self._target(ctx)),
            ("duplicates", checks.duplicates, lambda: self._duplicates(ctx)),
            ("identifiers", checks.identifiers, lambda: self._identifiers(ctx, reference_ids)),
        ]
        results: list[CheckResult] = []
        for name, enabled, run in groups:
            if not enabled:
                results.append(_skip(name, f"disabled in configuration (validation.checks.{name})"))
                continue
            try:
                results += run()
            except Exception as exc:  # fail closed: a crashing check must never look like a pass
                results.append(CheckResult(
                    f"{name}_internal_error", CheckStatus.FAIL, None, reason=(
                        f"the '{name}' checks could not be completed: {type(exc).__name__}: {exc}")))
        return ValidationReport(profile, ctx.n_rows, int(df.shape[1]), tuple(results))

    # ------------------------------------------------------------- schema
    def _schema(self, ctx: _Context) -> list[CheckResult]:
        out: list[CheckResult] = []
        dup = sorted(map(str, ctx.duplicate_labels))
        out.append(_result("duplicate_column_names", not dup, None, dup, [],
                           f"column name(s) appear more than once: {dup}"))
        out.append(_result("row_count", ctx.n_rows >= ctx.profile.min_rows, None, ctx.n_rows,
                           {"min": ctx.profile.min_rows},
                           f"{ctx.n_rows} row(s), fewer than the minimum {ctx.profile.min_rows}"))
        for col in ctx.required:
            name = "target_present" if ctx.is_target(col) else "column_present"
            here = col in ctx.labels
            out.append(_result(name, here, col, here, True, f"required column {col!r} is missing"))
        unexpected = [str(c) for c in ctx.df.columns if c not in ctx.role_of]
        if self._cfg.data_contract.allow_unexpected_columns:
            out.append(_skip("unexpected_columns", "allowed by data_contract.allow_unexpected_columns"))
        elif not unexpected:
            out.append(_result("unexpected_columns", True, None, [], [], ""))
        else:
            for col in dict.fromkeys(unexpected):
                out.append(_result("unexpected_columns", False, col, True, False,
                                   f"column {col!r} is not part of the '{ctx.profile_name}' profile"))
        for col in sorted(ctx.present):
            if col not in ctx.specs:
                continue
            problem = ctx.dtype_problem[col]
            out.append(_result("dtype", problem is None, col, str(ctx.series(col).dtype),
                               ctx.specs[col].dtype, problem or ""))
        return out

    # ----------------------------------------------------- missing values
    def _missing(self, ctx: _Context) -> list[CheckResult]:
        if ctx.n_rows == 0:
            return [_skip("missing_fraction", "no rows to evaluate")]
        limits = ctx.v.missing_values
        out: list[CheckResult] = []
        total_missing = 0
        total_cells = 0
        for col in sorted(ctx.present):
            n_missing = int(ctx.series(col).isna().sum())
            total_missing += n_missing
            total_cells += ctx.n_rows
            if ctx.is_target(col):
                continue  # reported by the target checks
            spec = ctx.specs[col]
            limit = limits.max_fraction_per_column[col] if spec.nullable else 0.0
            frac = n_missing / ctx.n_rows
            out.append(_result("missing_fraction", frac <= limit, col, frac, {"max": limit},
                               f"{n_missing} of {ctx.n_rows} values missing ({_pct(frac)}), above the "
                               f"allowed {_pct(limit)}"))
        if total_cells:
            overall = total_missing / total_cells
            out.append(_result("missing_fraction_overall", overall <= limits.max_overall_fraction, None,
                               overall, {"max": limits.max_overall_fraction},
                               f"{total_missing} of {total_cells} cells missing ({_pct(overall)}), above the "
                               f"allowed {_pct(limits.max_overall_fraction)}"))
        return out

    # ------------------------------------------------------------- ranges
    def _numeric_columns(self, ctx: _Context) -> list[str]:
        return [c for c in sorted(ctx.present) if c in ctx.specs and ctx.specs[c].dtype in ("integer", "float")
                and ctx.specs[c].allowed_values is None and not ctx.is_target(c)]

    def _ranges(self, ctx: _Context) -> list[CheckResult]:
        out: list[CheckResult] = []
        for col in self._numeric_columns(ctx):
            out += self._range_results(ctx, col, "non_finite", "numeric_range")
        if not out:
            out.append(_skip("numeric_range", "no numeric columns to check"))
        return out

    def _range_results(self, ctx: _Context, col: str, finite_name: str, range_name: str) -> list[CheckResult]:
        if not ctx.numeric_ok(col):
            reason = f"column is not numeric ({ctx.dtype_problem.get(col)})"
            return [_skip(finite_name, reason, col), _skip(range_name, reason, col)]
        values = ctx.series(col).dropna().astype(float)
        n_inf = int(np.isinf(values).sum())
        out = [_result(finite_name, n_inf == 0, col, n_inf, 0, f"{n_inf} infinite value(s)")]
        lo, hi = _effective_range(ctx.specs[col], ctx.v.numeric_ranges.get(col))
        finite = values[np.isfinite(values)]
        below = int((finite < lo).sum()) if lo is not None else 0
        above = int((finite > hi).sum()) if hi is not None else 0
        obs = {"min": _py(finite.min()) if len(finite) else None,
               "max": _py(finite.max()) if len(finite) else None, "n_out_of_range": below + above}
        out.append(_result(range_name, below + above == 0, col, obs, {"min": lo, "max": hi},
                           f"{below} value(s) below {lo} and {above} above {hi} "
                           f"(observed min {obs['min']}, max {obs['max']})"))
        return out

    # -------------------------------------------------------- categories
    def _categories(self, ctx: _Context) -> list[CheckResult]:
        out: list[CheckResult] = []
        for col in sorted(ctx.present):
            spec = ctx.specs.get(col)
            if spec is None or spec.allowed_values is None:
                continue
            if not ctx.dtype_ok(col):
                out.append(_skip("allowed_values", f"dtype check failed ({ctx.dtype_problem.get(col)})", col))
                continue
            values = ctx.series(col).dropna()
            if isinstance(values.dtype, pd.CategoricalDtype):
                values = values.astype(object)
            bad = values[~values.isin(spec.allowed_values)]
            out.append(_result(
                "allowed_values", bad.empty, col,
                {"n_invalid": int(len(bad)), "examples": _examples(bad.unique())}, list(spec.allowed_values),
                f"{len(bad)} value(s) outside the allowed set, e.g. {_examples(bad.unique())}"))
        return out

    # ------------------------------------------------------------- target
    def _target(self, ctx: _Context) -> list[CheckResult]:
        col = ctx.cfg.target
        if col not in ctx.role_of:
            return [_skip("target", "the target is not part of this profile's data")]
        if col not in ctx.present:
            return [_skip("target", f"target column {col!r} is absent (see target_present)", col)]
        if ctx.n_rows == 0:
            return [_skip("target", "no rows to evaluate", col)]
        n_missing = int(ctx.series(col).isna().sum())
        out = [_result("target_not_missing", n_missing == 0, col, n_missing, 0,
                       f"{n_missing} of {ctx.n_rows} target values are missing")]
        out += self._range_results(ctx, col, "target_finite", "target_range")
        if ctx.numeric_ok(col):
            finite = ctx.series(col).dropna().astype(float)
            finite = finite[np.isfinite(finite)]
            n_distinct = int(finite.nunique())
            need = ctx.v.target.min_distinct_values
            out.append(_result("target_not_constant", n_distinct >= need, col, n_distinct, {"min": need},
                               f"only {n_distinct} distinct target value(s); at least {need} required"))
        else:
            out.append(_skip("target_not_constant", f"column is not numeric ({ctx.dtype_problem.get(col)})", col))
        return out

    # --------------------------------------------------------- duplicates
    def _duplicates(self, ctx: _Context) -> list[CheckResult]:
        if not ctx.profile.check_duplicate_records:
            return [_skip("duplicate_records", f"not applicable to the '{ctx.profile_name}' profile")]
        if ctx.n_rows == 0:
            return [_skip("duplicate_records", "no rows to evaluate")]
        roles = ctx.v.duplicates.compare_roles
        cols = [c for role in roles for c in ctx.cfg.columns_for_role(role) if c in ctx.present]
        if not cols:
            return [_skip("duplicate_records", "none of the compared columns are present")]
        n_dup = int(ctx.df.duplicated(subset=cols).sum())
        frac = n_dup / ctx.n_rows
        limit = ctx.v.duplicates.max_duplicate_fraction
        return [_result("duplicate_records", frac <= limit, None, frac, {"max": limit},
                        f"{n_dup} record(s) duplicate an earlier one on {len(cols)} compared columns "
                        f"({_pct(frac)}), above the allowed {_pct(limit)}")]

    # -------------------------------------------------------- identifiers
    def _identifiers(self, ctx: _Context, reference_ids: Iterable[Any] | None) -> list[CheckResult]:
        out: list[CheckResult] = []
        for col in sorted(ctx.present):
            spec = ctx.specs.get(col)
            if spec is not None and spec.unique:
                n_dup = int(ctx.series(col).dropna().duplicated().sum())
                out.append(_result("unique_values", n_dup == 0, col, n_dup, 0,
                                   f"{n_dup} duplicated value(s) in a column that must be unique"))
        ident = ctx.cfg.identifier_column
        usable = ident in ctx.present and ctx.dtype_ok(ident)
        pattern = ctx.v.identifiers.pattern
        if pattern is not None:
            if not usable:
                out.append(_skip("identifier_format", "identifier column absent or failed its dtype check", ident))
            else:
                values = ctx.series(ident).dropna().astype(str)
                bad = values[~values.str.fullmatch(pattern)]
                out.append(_result("identifier_format", bad.empty, ident,
                                   {"n_invalid": int(len(bad)), "examples": _examples(bad.unique())}, pattern,
                                   f"{len(bad)} identifier(s) do not match {pattern!r}, e.g. {_examples(bad.unique())}"))
        if not ctx.profile.identifier_disjoint_from_training:
            out.append(_skip("identifier_disjoint_from_training",
                             f"not required by the '{ctx.profile_name}' profile"))
        elif ident not in ctx.present:
            out.append(_skip("identifier_disjoint_from_training", "identifier column absent", ident))
        elif reference_ids is None:
            out.append(CheckResult("identifier_disjoint_from_training", CheckStatus.FAIL, ident, None, 0,
                                   "training identifiers were not supplied, so overlap cannot be ruled out"))
        else:
            ref = set(pd.Series(list(reference_ids)).dropna())
            overlap = ctx.series(ident).dropna()
            overlap = overlap[overlap.isin(ref)]
            out.append(_result("identifier_disjoint_from_training", overlap.empty, ident,
                               {"n_overlap": int(len(overlap)), "examples": _examples(overlap.unique())}, 0,
                               f"{len(overlap)} identifier(s) already present in the training data, "
                               f"e.g. {_examples(overlap.unique())}"))
        return out


def validate_dataframe(
    df: pd.DataFrame,
    cfg: AppConfig,
    profile: str,
    *,
    reference_ids: Iterable[Any] | None = None,
) -> ValidationReport:
    """One-shot helper: build a validator from ``cfg`` and validate ``df``."""
    return DataValidator(cfg).validate(df, profile, reference_ids=reference_ids)
