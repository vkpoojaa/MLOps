from __future__ import annotations

import copy
import shutil
from pathlib import Path
from typing import Callable

import pandas as pd
import pytest
import yaml

from mlops_pipeline.config import load_config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def settings():
    return load_config(CONFIG_PATH, check_paths=True)


@pytest.fixture(scope="session")
def raw_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


@pytest.fixture
def write_config(tmp_path: Path, raw_config: dict) -> Callable[..., Path]:
    """Write a (mutated) copy of the real config to tmp_path/config/config.yaml."""

    def _write(mutate: Callable[[dict], None] | None = None) -> Path:
        data = copy.deepcopy(raw_config)
        if mutate:
            mutate(data)
        path = tmp_path / "config" / "config.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        return path

    return _write


@pytest.fixture
def valid_train_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "House_ID": ["HS-1", "HS-2", "HS-3"],
            "Area_sqft": [500, 900, 1200],
            "Bedrooms": [1, 2, 3],
            "Bathrooms": [1, 2, 2],
            "Age_Years": [10.0, None, 30.0],
            "Distance_to_Center_km": [3.5, 8.0, None],
            "Neighborhood_Score": [5.0, None, 9.5],
            "Has_Parking": [0, 1, 1],
            "Property_Condition": ["Poor", "Good", "Excellent"],
            "City_Zone": ["North", "Central", "West"],
            "Price_Lakhs": [20.5, 50.0, 80.25],
        }
    )


@pytest.fixture
def valid_incoming_df(valid_train_df: pd.DataFrame) -> pd.DataFrame:
    df = valid_train_df.copy()
    df.insert(1, "Batch", [1, 2, 3])
    return df
