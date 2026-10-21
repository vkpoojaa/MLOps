import pytest
import yaml

from mlops_pipeline.config import ConfigError, load_config
from mlops_pipeline.config.loader import CONFIG_ENV_VAR


def test_default_config_loads(settings):
    cfg = settings.config
    assert cfg.project.name == "mlops-house-price-pipeline"
    assert cfg.config_version == 1


def test_known_dataset_facts_come_from_yaml(settings):
    cfg = settings.config
    assert cfg.target == "Price_Lakhs"
    assert cfg.identifier_column == "House_ID"
    assert cfg.batch_column == "Batch"
    assert set(cfg.excluded_columns) == {"House_ID", "Batch"}
    assert cfg.batch_values == [1, 2, 3]
    assert set(cfg.categorical_features) == {"Property_Condition", "City_Zone"}
    assert len(cfg.feature_columns) == 9
    assert not {cfg.target, *cfg.excluded_columns} & set(cfg.feature_columns)


def test_paths_resolve_to_existing_files(settings, project_root):
    assert settings.project_root == project_root
    assert settings.train_path.is_file()
    assert settings.incoming_path.is_file()


def test_model_gate_thresholds_are_resolved_from_config(settings):
    pending = set(settings.pending)
    assert "model_gate.quality_threshold" not in pending
    assert "model_gate.production_comparison.min_improvement" not in pending
    assert settings.config.model_gate.quality_threshold == pytest.approx(5.793117272533898)
    assert settings.config.model_gate.production_comparison.min_improvement == 0.0
    assert settings.config.model_gate.production_metadata_path == "artifacts/models/production_model_metadata.json"
    # Phase 3A decided the data-validation settings (rationale: docs/DATA_VALIDATION.md)
    assert not {s for s in pending if s.startswith("validation.")}
    # Phase 2 resolved these modelling settings; they must no longer be pending
    assert not {s for s in pending if s.startswith("model.") and s != "model.selected"}


def test_missing_file_raises(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.yaml")


def test_invalid_yaml_raises(tmp_path):
    bad = tmp_path / "config" / "config.yaml"
    bad.parent.mkdir()
    bad.write_text("a: [unclosed", encoding="utf-8")
    with pytest.raises(ConfigError, match="YAML parse error"):
        load_config(bad)


def test_non_mapping_top_level_raises(tmp_path):
    bad = tmp_path / "config" / "config.yaml"
    bad.parent.mkdir()
    bad.write_text("- just\n- a list\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="mapping"):
        load_config(bad)


def test_duplicate_keys_rejected(tmp_path, raw_config):
    text = yaml.safe_dump(raw_config, sort_keys=False) + "config_version: 2\n"
    path = tmp_path / "config" / "config.yaml"
    path.parent.mkdir()
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError, match="duplicate key"):
        load_config(path)


def test_unknown_key_rejected(write_config):
    path = write_config(lambda d: d["model"].update(algoritm="typo"))
    with pytest.raises(ConfigError, match="algoritm"):
        load_config(path)


def test_missing_required_key_rejected(write_config):
    path = write_config(lambda d: d["model_gate"].pop("quality_threshold"))
    with pytest.raises(ConfigError, match="quality_threshold"):
        load_config(path)


def test_env_var_selects_config(write_config, monkeypatch):
    path = write_config()
    monkeypatch.setenv(CONFIG_ENV_VAR, str(path))
    assert load_config().config_path == path.resolve()


def test_check_paths_detects_missing_data(write_config):
    path = write_config()  # tmp repo has no data/ directory
    load_config(path)  # fine without the path check
    with pytest.raises(ConfigError, match="file not found"):
        load_config(path, check_paths=True)


def test_model_gate_thresholds_must_be_valid_and_non_pending(write_config):
    bad_quality = write_config(lambda d: d["model_gate"].update(quality_threshold=None))
    with pytest.raises(ConfigError, match="quality_threshold"):
        load_config(bad_quality)

    bad_improvement = write_config(
        lambda d: d["model_gate"]["production_comparison"].update(min_improvement=-0.1)
    )
    with pytest.raises(ConfigError, match="min_improvement"):
        load_config(bad_improvement)
