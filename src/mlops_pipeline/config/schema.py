"""Typed schema for ``config/config.yaml``.

Structure and types only; cross-section rules live in ``validation.py``.
Unknown keys are rejected everywhere (``extra="forbid"``) so typos fail on load.
Fields typed ``X | None`` without a default are *required keys* whose value may
be ``null`` (= decision pending); omitting the key is an error.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, model_validator

DatasetName = Literal["train", "incoming"]
Dtype = Literal["integer", "float", "string", "categorical"]


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# ----------------------------------------------------------------- basics
class ProjectSection(_Base):
    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    description: str


class ReproducibilitySection(_Base):
    random_seed: int = Field(ge=0)
    metric_tolerance: float = Field(ge=0, le=0.01)


class PathsSection(_Base):
    train_data: str
    incoming_data: str
    interim_dir: str
    processed_dir: str
    artifacts_dir: str
    logs_dir: str
    reports_dir: str
    models_dir: str
    selected_model: str
    evaluation_report: str


# ---------------------------------------------------------------- columns
class FeaturesSection(_Base):
    numeric: list[str]
    categorical: list[str]


class ExcludedColumn(_Base):
    role: Literal["identifier", "batch"]
    reason: str = Field(min_length=1)


class ColumnsSection(_Base):
    target: str = Field(min_length=1)
    features: FeaturesSection
    excluded: dict[str, ExcludedColumn]


# ---------------------------------------------------------- data contract
class ColumnSpec(_Base):
    dtype: Dtype
    nullable: bool
    available_in: list[DatasetName] = Field(min_length=1)
    unique: bool = False
    allowed_values: list[StrictInt | StrictStr] | None = None
    min: float | None = None
    max: float | None = None

    @model_validator(mode="after")
    def _check_spec(self) -> "ColumnSpec":
        numeric = self.dtype in ("integer", "float")
        if self.dtype == "categorical" and not self.allowed_values:
            raise ValueError("categorical columns must define allowed_values")
        if (self.min is not None or self.max is not None) and not numeric:
            raise ValueError("min/max are only valid for numeric dtypes")
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError(f"min ({self.min}) must be <= max ({self.max})")
        if self.allowed_values:
            kinds = {type(v) for v in self.allowed_values}
            if len(kinds) > 1:
                raise ValueError("allowed_values must all be the same type")
            if len(set(self.allowed_values)) != len(self.allowed_values):
                raise ValueError("allowed_values contains duplicates")
            is_text = kinds == {str}
            if numeric and is_text or (not numeric and not is_text):
                raise ValueError("allowed_values type does not match dtype")
        if len(set(self.available_in)) != len(self.available_in):
            raise ValueError("available_in contains duplicates")
        return self


class DataContractSection(_Base):
    version: int = Field(ge=1)
    allow_unexpected_columns: bool
    columns: dict[str, ColumnSpec] = Field(min_length=1)


# ---------------------------------------------------------- preprocessing
class ImputationSettings(_Base):
    strategy: Literal["mean", "median", "most_frequent"]
    add_missing_indicator: bool


class NumericPreprocessing(_Base):
    imputation: ImputationSettings
    scaling: Literal["none", "standard", "minmax"]


class EncoderSpec(_Base):
    type: Literal["ordinal", "onehot"]
    order: list[str] | None = None

    @model_validator(mode="after")
    def _check_order(self) -> "EncoderSpec":
        if self.type == "ordinal" and not self.order:
            raise ValueError("ordinal encoders require a non-empty order")
        if self.type == "onehot" and self.order is not None:
            raise ValueError("onehot encoders must not define order")
        return self


class CategoricalPreprocessing(_Base):
    handle_unknown: Literal["ignore", "error"]
    encoders: dict[str, EncoderSpec]


class PreprocessingSection(_Base):
    numeric: NumericPreprocessing
    categorical: CategoricalPreprocessing


# ------------------------------------------------------------------ model
Algorithm = Literal["linear_regression", "ridge", "random_forest", "gradient_boosting"]
MetricName = Literal["mae", "rmse", "r2"]
ModelRole = Literal["baseline", "candidate"]


class ModelSpec(_Base):
    name: str = Field(min_length=1)
    algorithm: Algorithm
    hyperparameters: dict[str, object]


class SplitSettings(_Base):
    test_size: float = Field(gt=0.0, lt=1.0)


class EvaluationSettings(_Base):
    primary_metric: MetricName
    secondary_metrics: list[MetricName]
    cv_folds: int = Field(ge=2)


class ModelSection(_Base):
    problem_type: Literal["regression"]
    baseline: ModelSpec
    candidate: ModelSpec
    selected: ModelRole | None  # null = decision pending until results are reviewed
    split: SplitSettings
    evaluation: EvaluationSettings


# ------------------------------------------------------------- data gate
Role = Literal["features", "target", "identifier", "batch"]
ProfileName = Literal["train", "incoming", "inference"]


class ValidationChecks(_Base):
    schema_: bool = Field(alias="schema")
    value_ranges: bool
    missing_values: bool
    categorical_values: bool
    target: bool
    duplicates: bool
    identifiers: bool

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)


class MissingValueLimits(_Base):
    # Required: every nullable column needs an explicit limit (checked in validation.py).
    max_overall_fraction: float = Field(ge=0.0, le=1.0)
    max_fraction_per_column: dict[str, float]

    @model_validator(mode="after")
    def _check_fractions(self) -> "MissingValueLimits":
        for col, frac in self.max_fraction_per_column.items():
            if not 0.0 <= frac <= 1.0:
                raise ValueError(f"max_fraction_per_column[{col!r}] must be in [0, 1]")
        return self


class NumericRange(_Base):
    min: float | None = None
    max: float | None = None

    @model_validator(mode="after")
    def _check_range(self) -> "NumericRange":
        if self.min is None and self.max is None:
            raise ValueError("a range needs at least one of min / max")
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError(f"min ({self.min}) must be <= max ({self.max})")
        return self


class TargetRules(_Base):
    min_distinct_values: int = Field(ge=2)


class DuplicateRules(_Base):
    compare_roles: list[Literal["features", "target"]] = Field(min_length=1)
    max_duplicate_fraction: float = Field(ge=0.0, le=1.0)


class IdentifierRules(_Base):
    pattern: str | None  # regular expression the identifier must fully match; null = no format check


class DatasetProfile(_Base):
    contract_dataset: DatasetName
    required_roles: list[Role]
    optional_roles: list[Role]
    min_rows: int = Field(ge=1)
    check_duplicate_records: bool
    identifier_disjoint_from_training: bool


class ValidationSection(_Base):
    checks: ValidationChecks
    missing_values: MissingValueLimits
    numeric_ranges: dict[str, NumericRange]
    target: TargetRules
    duplicates: DuplicateRules
    identifiers: IdentifierRules
    profiles: dict[ProfileName, DatasetProfile] = Field(min_length=1)
    on_failure: Literal["halt"]


# ------------------------------------------------------------- model gate
class ProductionComparison(_Base):
    enabled: bool
    min_improvement: float = Field(ge=0.0, allow_inf_nan=False)


class ModelGateSection(_Base):
    quality_threshold: float = Field(gt=0.0, allow_inf_nan=False)
    production_metadata_path: str = Field(min_length=1)
    production_comparison: ProductionComparison
    on_failure: Literal["halt"]


# ------------------------------------------------------------- monitoring
class DataDriftSettings(_Base):
    method: str | None
    bootstrap_runs: int = Field(gt=0)
    bootstrap_batch_size: int = Field(gt=0)
    per_feature_threshold: float | None
    drifted_feature_share_threshold: float | None = Field(ge=0.0, le=1.0)


class PerformanceDriftSettings(_Base):
    bootstrap_runs: int = Field(gt=0)
    bootstrap_batch_size: int = Field(gt=0)
    max_degradation: float | None


class OperationalSettings(_Base):
    latency_p95_ms_max: float | None = Field(gt=0.0)
    error_rate_max: float | None = Field(ge=0.0, le=1.0)


class MonitoringSection(_Base):
    reference_dataset: DatasetName
    data_drift: DataDriftSettings
    performance_drift: PerformanceDriftSettings
    operational: OperationalSettings
    on_alert: str | None


# ------------------------------------------------------------- experiment tracking
class MLflowSection(_Base):
    enabled: bool
    tracking_uri: str = Field(min_length=1)
    experiment_name: str = Field(min_length=1)
    artifact_subdirectory: str = Field(min_length=1)


class ModelRegistrySection(_Base):
    enabled: bool
    registered_model_name: str = Field(min_length=1)
    candidate_alias: str = Field(min_length=1)
    production_alias: str = Field(min_length=1)
    await_registration_for_seconds: int = Field(ge=0, le=3600)

    @model_validator(mode="after")
    def _check_aliases(self) -> "ModelRegistrySection":
        if self.candidate_alias == self.production_alias:
            raise ValueError("candidate_alias and production_alias must differ")
        return self


# ------------------------------------------------------------------- root
class AppConfig(_Base):
    config_version: int = Field(ge=1)
    project: ProjectSection
    reproducibility: ReproducibilitySection
    paths: PathsSection
    columns: ColumnsSection
    data_contract: DataContractSection
    preprocessing: PreprocessingSection
    model: ModelSection
    validation: ValidationSection
    model_gate: ModelGateSection
    monitoring: MonitoringSection
    mlflow: MLflowSection
    model_registry: ModelRegistrySection

    # ---- values derived from the YAML (never hard-coded) -----------------
    @property
    def target(self) -> str:
        return self.columns.target

    @property
    def numeric_features(self) -> list[str]:
        return list(self.columns.features.numeric)

    @property
    def categorical_features(self) -> list[str]:
        return list(self.columns.features.categorical)

    @property
    def feature_columns(self) -> list[str]:
        return self.numeric_features + self.categorical_features

    @property
    def excluded_columns(self) -> list[str]:
        return list(self.columns.excluded)

    def _excluded_with_role(self, role: str) -> str:
        matches = [c for c, spec in self.columns.excluded.items() if spec.role == role]
        if len(matches) != 1:
            raise ValueError(f"expected exactly one excluded column with role {role!r}")
        return matches[0]

    @property
    def identifier_column(self) -> str:
        return self._excluded_with_role("identifier")

    @property
    def batch_column(self) -> str:
        return self._excluded_with_role("batch")

    def columns_for_role(self, role: str) -> list[str]:
        """Column names that play ``role`` (features | target | identifier | batch)."""
        if role == "features":
            return self.feature_columns
        if role == "target":
            return [self.target]
        if role == "identifier":
            return [self.identifier_column]
        if role == "batch":
            return [self.batch_column]
        raise ValueError(f"unknown column role {role!r}")

    @property
    def random_seed(self) -> int:
        return self.reproducibility.random_seed

    @property
    def metric_names(self) -> list[str]:
        """Primary metric first, then the secondary ones."""
        ev = self.model.evaluation
        return [ev.primary_metric, *ev.secondary_metrics]

    def model_spec(self, role: ModelRole) -> ModelSpec:
        return getattr(self.model, role)

    @property
    def batch_values(self) -> list[int]:
        allowed = self.data_contract.columns[self.batch_column].allowed_values or []
        return [int(v) for v in allowed]
