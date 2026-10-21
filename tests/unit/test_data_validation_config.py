"""Invalid / incomplete / contradictory validation configuration must fail loudly.

Two layers are tested: ``load_config`` (typed schema + cross-section rules) and
``DataValidator(...)`` (re-checks settings, so an ``AppConfig`` that bypassed
loading still cannot reach a running gate). Column names are derived from the
config, not hard-coded.
"""

from __future__ import annotations

import pytest

from mlops_pipeline.config import ConfigError, load_config
from mlops_pipeline.config.validation import check_data_validation_settings
from mlops_pipeline.data import DataValidator


def _fails(write_config, mutate, match):
    with pytest.raises(ConfigError, match=match):
        load_config(write_config(mutate))


def _nullable(cfg):
    return next(c for c in cfg.feature_columns if cfg.data_contract.columns[c].nullable)


def _non_nullable(cfg):
    return next(c for c in cfg.feature_columns if not cfg.data_contract.columns[c].nullable)


def _ranged(cfg):
    """A numeric feature whose range lives in validation.numeric_ranges (not the contract)."""
    return next(c for c in cfg.validation.numeric_ranges if c in cfg.numeric_features)


def test_shipped_validation_config_is_consistent(settings):
    assert check_data_validation_settings(settings.config) == []
    assert not [p for p in settings.pending if p.startswith("validation.")]


def test_validator_can_be_built_from_valid_config(settings):
    assert set(DataValidator.from_settings(settings).profiles) == {"train", "incoming", "inference"}


# ----------------------------------------------------- missing / null settings
def test_null_overall_missing_limit_is_rejected(write_config):
    _fails(write_config, lambda d: d["validation"]["missing_values"].update(max_overall_fraction=None),
           "max_overall_fraction")


def test_missing_per_column_mapping_is_rejected(write_config):
    _fails(write_config, lambda d: d["validation"]["missing_values"].pop("max_fraction_per_column"),
           "max_fraction_per_column")


def test_null_per_column_mapping_is_rejected(write_config):
    _fails(write_config, lambda d: d["validation"]["missing_values"].update(max_fraction_per_column=None),
           "max_fraction_per_column")


@pytest.mark.parametrize("key", ["checks", "numeric_ranges", "target", "duplicates", "identifiers", "profiles"])
def test_required_validation_subsections_cannot_be_omitted(write_config, key):
    _fails(write_config, lambda d: d["validation"].pop(key), key)


def test_nullable_column_without_a_limit_is_rejected(write_config, settings):
    col = _nullable(settings.config)
    _fails(write_config, lambda d: d["validation"]["missing_values"]["max_fraction_per_column"].pop(col),
           f"no limit for nullable column '{col}'")


def test_numeric_column_without_a_range_is_rejected(write_config, settings):
    col = _ranged(settings.config)
    _fails(write_config, lambda d: d["validation"]["numeric_ranges"].pop(col), f"'{col}' has no complete")


def test_one_sided_range_is_rejected_when_contract_gives_no_other_bound(write_config, settings):
    col = _ranged(settings.config)
    _fails(write_config, lambda d: d["validation"]["numeric_ranges"][col].pop("max"), f"'{col}' has no complete")


def test_unknown_key_in_validation_section_is_rejected(write_config):
    _fails(write_config, lambda d: d["validation"].update(typo_limit=1), "typo_limit")


# ------------------------------------------------------------ malformed values
@pytest.mark.parametrize("bad", [-0.01, 1.01, "ten percent"])
def test_overall_fraction_must_be_a_fraction(write_config, bad):
    _fails(write_config, lambda d: d["validation"]["missing_values"].update(max_overall_fraction=bad),
           "max_overall_fraction")


@pytest.mark.parametrize("bad", [-0.1, 1.5])
def test_per_column_fraction_must_be_a_fraction(write_config, settings, bad):
    col = _nullable(settings.config)
    _fails(write_config, lambda d: d["validation"]["missing_values"]["max_fraction_per_column"].update({col: bad}),
           "must be in")


def test_range_with_min_above_max_is_rejected(write_config, settings):
    col = _ranged(settings.config)
    _fails(write_config, lambda d: d["validation"]["numeric_ranges"][col].update(min=10, max=1), "must be <= max")


def test_empty_range_is_rejected(write_config, settings):
    col = _ranged(settings.config)
    _fails(write_config, lambda d: d["validation"]["numeric_ranges"].update({col: {}}), "at least one of min")


def test_invalid_identifier_pattern_is_rejected(write_config):
    _fails(write_config, lambda d: d["validation"]["identifiers"].update(pattern="([unclosed"),
           "invalid regular expression")


def test_target_min_distinct_values_must_be_at_least_two(write_config):
    _fails(write_config, lambda d: d["validation"]["target"].update(min_distinct_values=1), "min_distinct_values")


def test_unknown_profile_name_is_rejected(write_config):
    _fails(write_config, lambda d: d["validation"]["profiles"].update(staging=d["validation"]["profiles"]["train"]),
           "profiles")


def test_unknown_role_is_rejected(write_config):
    _fails(write_config, lambda d: d["validation"]["profiles"]["train"]["required_roles"].append("price"),
           "required_roles")


def test_min_rows_must_be_positive(write_config):
    _fails(write_config, lambda d: d["validation"]["profiles"]["train"].update(min_rows=0), "min_rows")


# ------------------------------------------------------------- contradictions
def test_missing_limit_for_unknown_column_is_rejected(write_config):
    _fails(write_config,
           lambda d: d["validation"]["missing_values"]["max_fraction_per_column"].update(No_Such_Column=0.1),
           "not a data_contract column")


def test_positive_missing_limit_for_non_nullable_column_contradicts_contract(write_config, settings):
    col = _non_nullable(settings.config)
    _fails(write_config,
           lambda d: d["validation"]["missing_values"]["max_fraction_per_column"].update({col: 0.1}),
           "contradicts")


def test_overall_zero_with_positive_column_limits_is_contradictory(write_config):
    _fails(write_config, lambda d: d["validation"]["missing_values"].update(max_overall_fraction=0.0),
           "max_overall_fraction is 0 but")


def test_range_for_unknown_or_non_numeric_column_is_rejected(write_config, settings):
    cat = settings.config.categorical_features[0]
    _fails(write_config, lambda d: d["validation"]["numeric_ranges"].update({cat: {"min": 0, "max": 1}}),
           "not numeric")
    _fails(write_config, lambda d: d["validation"]["numeric_ranges"].update(No_Such_Column={"min": 0, "max": 1}),
           "not a data_contract column")


def test_range_for_column_with_allowed_values_is_rejected(write_config, settings):
    cfg = settings.config
    col = next(c for c in cfg.numeric_features if cfg.data_contract.columns[c].allowed_values is not None)
    _fails(write_config, lambda d: d["validation"]["numeric_ranges"].update({col: {"min": 0, "max": 1}}),
           "already has allowed_values")


def test_range_wider_than_the_contract_documented_range_is_rejected(write_config, settings):
    cfg = settings.config
    col = next(c for c in cfg.numeric_features if cfg.data_contract.columns[c].min is not None)
    spec = cfg.data_contract.columns[col]
    _fails(write_config,
           lambda d: d["validation"]["numeric_ranges"].update({col: {"min": spec.min - 1, "max": spec.max}}),
           "wider than the contract min")
    _fails(write_config,
           lambda d: d["validation"]["numeric_ranges"].update({col: {"min": spec.min, "max": spec.max + 1}}),
           "wider than the contract max")


def test_training_profile_must_require_the_target(write_config):
    _fails(write_config, lambda d: d["validation"]["profiles"]["train"].update(
        required_roles=["identifier", "features"], optional_roles=["target"]), "must require the 'target'")


def test_profile_must_require_features(write_config):
    _fails(write_config, lambda d: d["validation"]["profiles"]["inference"].update(required_roles=["identifier"],
                                                                                   optional_roles=[]),
           "must include 'features'")


def test_role_cannot_be_both_required_and_optional(write_config):
    _fails(write_config, lambda d: d["validation"]["profiles"]["inference"]["optional_roles"].append("features"),
           "both required and optional")


def test_profile_cannot_require_a_column_missing_from_its_contract_dataset(write_config, settings):
    _fails(write_config, lambda d: d["validation"]["profiles"]["train"]["required_roles"].append("batch"),
           "not part of the 'train' data contract")


def test_identifier_disjointness_is_meaningless_for_the_training_data(write_config):
    _fails(write_config,
           lambda d: d["validation"]["profiles"]["train"].update(identifier_disjoint_from_training=True),
           "makes no sense for the training data")


def test_identifier_disjointness_needs_the_identifier_role(write_config):
    _fails(write_config, lambda d: d["validation"]["profiles"]["inference"].update(
        identifier_disjoint_from_training=True, optional_roles=["target"]), "needs the 'identifier' role")


def test_dependent_checks_cannot_run_without_the_schema_check(write_config):
    _fails(write_config, lambda d: d["validation"]["checks"].update(schema=False), "'schema' is disabled")


def test_all_checks_may_be_disabled_together(write_config):
    def off(d):
        for key in d["validation"]["checks"]:
            d["validation"]["checks"][key] = False
    assert load_config(write_config(off)).config.validation.checks.schema_ is False


def test_on_failure_only_allows_halt(write_config):
    _fails(write_config, lambda d: d["validation"].update(on_failure="continue"), "on_failure")


# --------------------------------------------- validator re-checks bypassed config
def test_validator_rejects_a_config_that_bypassed_loading(settings):
    cfg = settings.config
    col = _nullable(cfg)
    limits = {k: v for k, v in cfg.validation.missing_values.max_fraction_per_column.items() if k != col}
    mv = cfg.validation.missing_values.model_copy(update={"max_fraction_per_column": limits})
    broken = cfg.model_copy(update={"validation": cfg.validation.model_copy(update={"missing_values": mv})})
    with pytest.raises(ConfigError, match="no limit for nullable column"):
        DataValidator(broken)


def test_config_error_reports_all_problems_at_once(write_config, settings):
    cfg = settings.config
    col, other = _nullable(cfg), _non_nullable(cfg)

    def mutate(d):
        d["validation"]["missing_values"]["max_fraction_per_column"].pop(col)
        d["validation"]["missing_values"]["max_fraction_per_column"][other] = 0.5
        d["validation"]["identifiers"]["pattern"] = "(["

    with pytest.raises(ConfigError) as exc:
        load_config(write_config(mutate))
    assert len(exc.value.problems) >= 3
