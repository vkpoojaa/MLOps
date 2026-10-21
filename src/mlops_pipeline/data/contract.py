"""Data contract: what a raw dataset must look like.

The contract is *defined in the YAML* (``data_contract`` section); this module
only interprets it. ``DataContract.check`` reports structural conformance
(columns, dtypes, nullability, allowed values, documented ranges, uniqueness).

This is NOT the pipeline's data gate. Gate behaviour (missing-value limits,
halting a run, keeping the previous model) is a later phase. Missing counts of
nullable columns are reported for information only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from pandas.api import types as pdt

from ..config.schema import AppConfig, ColumnSpec, DatasetName


class ContractViolationError(Exception):
    """Raised by ``ContractReport.raise_if_failed``."""


@dataclass(frozen=True)
class Violation:
    rule: str
    message: str
    column: str | None = None
    count: int | None = None

    def __str__(self) -> str:
        where = f"[{self.column}] " if self.column else ""
        return f"{self.rule}: {where}{self.message}"


@dataclass(frozen=True)
class ContractReport:
    dataset: str
    n_rows: int
    violations: tuple[Violation, ...] = ()
    null_counts: dict[str, int] = field(default_factory=dict)  # informational only

    @property
    def ok(self) -> bool:
        return not self.violations

    def to_text(self) -> str:
        status = "CONFORMS" if self.ok else f"{len(self.violations)} VIOLATION(S)"
        lines = [f"Data contract check - {self.dataset} ({self.n_rows} rows): {status}"]
        lines += [f"  - {v}" for v in self.violations]
        nulls = {c: n for c, n in self.null_counts.items() if n}
        if nulls:
            lines.append("  missing values in nullable columns (informational): " + ", ".join(
                f"{c}={n}" for c, n in nulls.items()))
        return "\n".join(lines)

    def raise_if_failed(self) -> None:
        if not self.ok:
            raise ContractViolationError(self.to_text())


class DataContract:
    def __init__(self, columns: dict[str, ColumnSpec], *, allow_unexpected_columns: bool, version: int):
        self.columns = dict(columns)
        self.allow_unexpected_columns = allow_unexpected_columns
        self.version = version

    @classmethod
    def from_config(cls, cfg: AppConfig) -> "DataContract":
        c = cfg.data_contract
        return cls(c.columns, allow_unexpected_columns=c.allow_unexpected_columns, version=c.version)

    def columns_for(self, dataset: DatasetName) -> dict[str, ColumnSpec]:
        return {n: s for n, s in self.columns.items() if dataset in s.available_in}

    # ------------------------------------------------------------- checks
    def check(self, df: pd.DataFrame, dataset: DatasetName) -> ContractReport:
        expected = self.columns_for(dataset)
        violations: list[Violation] = []

        for name in expected:
            if name not in df.columns:
                violations.append(Violation("missing_column", "required column not found", name))
        if not self.allow_unexpected_columns:
            for name in df.columns:
                if name not in expected:
                    violations.append(Violation(
                        "unexpected_column", f"not part of the {dataset} contract", str(name)))

        null_counts: dict[str, int] = {}
        for name, spec in expected.items():
            if name not in df.columns:
                continue
            series = df[name]
            n_null = int(series.isna().sum())
            if spec.nullable:
                null_counts[name] = n_null
            elif n_null:
                violations.append(Violation("null_not_allowed", "column is not nullable", name, n_null))

            dtype_problem = _dtype_problem(series, spec)
            if dtype_problem:
                violations.append(Violation("dtype", dtype_problem, name))
                continue  # value checks are meaningless on the wrong dtype
            violations += _value_violations(name, series, spec)

        return ContractReport(dataset, len(df), tuple(violations), null_counts)


# ------------------------------------------------------------------ helpers
def _dtype_problem(series: pd.Series, spec: ColumnSpec) -> str | None:
    s = series.astype(object) if isinstance(series.dtype, pd.CategoricalDtype) else series
    if spec.dtype in ("string", "categorical"):
        # pandas may infer object dtype for a text column containing nulls.
        # Treat it as text when every non-null value is a string; missingness is
        # evaluated by the dedicated nullability/quality checks.
        if pdt.is_string_dtype(s) or all(isinstance(v, str) for v in s.dropna().tolist()):
            return None
        return f"expected text, got {series.dtype}"
    if pdt.is_bool_dtype(s) or not pdt.is_numeric_dtype(s):
        return f"expected numeric, got {series.dtype}"
    if spec.dtype == "integer" and not pdt.is_integer_dtype(s):
        values = s.dropna()
        if not pdt.is_float_dtype(s) or not spec.nullable or not ((values % 1) == 0).all():
            return f"expected integer values, got {series.dtype} with non-integer values"
    return None


def _value_violations(name: str, series: pd.Series, spec: ColumnSpec) -> list[Violation]:
    out: list[Violation] = []
    values = series.dropna()
    if spec.allowed_values is not None:
        bad = int((~values.isin(spec.allowed_values)).sum())
        if bad:
            out.append(Violation("allowed_values", f"values outside {list(spec.allowed_values)}", name, bad))
    if spec.min is not None:
        bad = int((values < spec.min).sum())
        if bad:
            out.append(Violation("min", f"values below {spec.min}", name, bad))
    if spec.max is not None:
        bad = int((values > spec.max).sum())
        if bad:
            out.append(Violation("max", f"values above {spec.max}", name, bad))
    if spec.unique:
        dup = int(values.duplicated().sum())
        if dup:
            out.append(Violation("unique", "duplicate values", name, dup))
    return out
