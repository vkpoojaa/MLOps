"""Load the historical training dataset (never the incoming batches).

The raw file is read and checked against the YAML data contract before anything
else touches it. Incoming batches are deliberately NOT loadable from here: the
initial model is trained on the historical file only.
"""

from __future__ import annotations

import pandas as pd

from ..config import Settings
from .contract import DataContract


def load_training_data(settings: Settings) -> pd.DataFrame:
    """Read the training CSV and require it to conform to the data contract.

    Raises ``ContractViolationError`` on structural problems and ``ValueError``
    if batch metadata is present (it only exists in the incoming file).
    """
    cfg = settings.config
    df = pd.read_csv(settings.train_path)
    if cfg.batch_column in df.columns:
        raise ValueError(
            f"{cfg.batch_column!r} found in the training file; batch metadata belongs to "
            "incoming data and the initial model must not see it"
        )
    DataContract.from_config(cfg).check(df, "train").raise_if_failed()
    return df
