import pytest

from mlops_pipeline.config import ConfigError, load_config


def _fails(write_config, mutate, match):
    with pytest.raises(ConfigError, match=match):
        load_config(write_config(mutate))


def test_target_cannot_be_a_feature(write_config):
    _fails(write_config, lambda d: d["columns"]["features"]["numeric"].append("Price_Lakhs"),
           "must not be a feature")


def test_excluded_column_cannot_be_a_feature(write_config):
    _fails(write_config, lambda d: d["columns"]["features"]["numeric"].append("House_ID"),
           "both a feature and an excluded")


def test_batch_cannot_be_a_feature(write_config):
    _fails(write_config, lambda d: d["columns"]["features"]["categorical"].append("Batch"),
           "both a feature and an excluded")


def test_duplicate_feature_rejected(write_config):
    _fails(write_config, lambda d: d["columns"]["features"]["numeric"].append("Area_sqft"),
           "more than once")


def test_unassigned_contract_column_rejected(write_config):
    def mutate(d):
        d["data_contract"]["columns"]["New_Col"] = {
            "dtype": "float", "nullable": False, "available_in": ["train", "incoming"]}
    _fails(write_config, mutate, "has no role")


def test_role_without_contract_entry_rejected(write_config):
    _fails(write_config, lambda d: d["data_contract"]["columns"].pop("Area_sqft"),
           "missing from data_contract")


def test_batch_must_be_incoming_only(write_config):
    _fails(write_config,
           lambda d: d["data_contract"]["columns"]["Batch"].update(available_in=["train", "incoming"]),
           "incoming only")


def test_feature_must_exist_in_both_datasets(write_config):
    _fails(write_config,
           lambda d: d["data_contract"]["columns"]["Area_sqft"].update(available_in=["train"]),
           "both train and incoming")


def test_target_must_be_float_and_not_nullable(write_config):
    _fails(write_config,
           lambda d: d["data_contract"]["columns"]["Price_Lakhs"].update(nullable=True),
           "must not be nullable")


def test_encoder_missing_for_categorical_feature(write_config):
    _fails(write_config,
           lambda d: d["preprocessing"]["categorical"]["encoders"].pop("City_Zone"),
           "no encoder for feature")


def test_ordinal_order_must_match_contract(write_config):
    _fails(write_config,
           lambda d: d["preprocessing"]["categorical"]["encoders"]["Property_Condition"].update(
               order=["Poor", "Fair", "Good"]),
           "must list exactly")


def test_contract_min_greater_than_max_rejected(write_config):
    _fails(write_config,
           lambda d: d["data_contract"]["columns"]["Neighborhood_Score"].update(min=11, max=10),
           "must be <= max")


def test_categorical_requires_allowed_values(write_config):
    _fails(write_config,
           lambda d: d["data_contract"]["columns"]["City_Zone"].pop("allowed_values"),
           "allowed_values")


@pytest.mark.parametrize("bad", ["/etc/data.csv", "../outside.csv", "C:\\data\\x.csv"])
def test_paths_must_be_relative_and_inside_repo(write_config, bad):
    _fails(write_config, lambda d: d["paths"].update(train_data=bad), "paths.train_data")


@pytest.mark.parametrize(
    "section,key,value",
    [
        (("validation", "missing_values"), "max_overall_fraction", 1.5),
        (("model", "split"), "test_size", 1.0),
        (("monitoring", "operational"), "error_rate_max", -0.1),
    ],
)
def test_out_of_range_thresholds_rejected_when_set(write_config, section, key, value):
    def mutate(d):
        node = d
        for part in section:
            node = node[part]
        node[key] = value
    _fails(write_config, mutate, key)


def test_valid_threshold_accepted_when_set(write_config):
    settings = load_config(write_config(
        lambda d: d["validation"]["missing_values"].update(max_overall_fraction=0.2)))
    assert settings.config.validation.missing_values.max_overall_fraction == 0.2
    assert "validation.missing_values.max_overall_fraction" not in settings.pending
