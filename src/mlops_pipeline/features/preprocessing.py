"""Build the preprocessing step from the YAML config.

The returned ``ColumnTransformer`` is *unfitted*. It must only ever be fitted on
training data, which is guaranteed by embedding it in a single sklearn
``Pipeline`` (see ``models.factory.build_pipeline``): the pipeline's ``fit`` is
the only place statistics (imputation values, category lists, scaling factors)
are learned, and cross-validation re-fits it inside every fold.

Design:
* Columns are selected BY NAME from config; ``remainder="drop"`` means any other
  column (identifier, batch, target) is discarded even if it is passed in.
* Numeric: imputation (strategy from config) then optional scaling.
* Ordinal categorical: explicit category order from config. Unknown categories
  become missing (when ``handle_unknown: ignore``) and are then imputed with the
  most frequent category seen in training; with ``error`` they raise.
* One-hot categorical: categories learned from training data; unknown categories
  at prediction time become all-zero rows (``ignore``) or raise (``error``).
"""

from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder, StandardScaler

from ..config.schema import AppConfig, EncoderSpec

NUMERIC_BLOCK = "numeric"


def _numeric_block(cfg: AppConfig) -> Pipeline:
    imp = cfg.preprocessing.numeric.imputation
    steps = [
        (
            "impute",
            SimpleImputer(strategy=imp.strategy, add_indicator=imp.add_missing_indicator),
        )
    ]
    scaling = cfg.preprocessing.numeric.scaling
    if scaling == "standard":
        steps.append(("scale", StandardScaler()))
    elif scaling == "minmax":
        steps.append(("scale", MinMaxScaler()))
    return Pipeline(steps)


def _categorical_block(spec: EncoderSpec, handle_unknown: str):
    if spec.type == "onehot":
        return OneHotEncoder(handle_unknown=handle_unknown, sparse_output=False)
    # ordinal with an explicit, configured order
    if handle_unknown == "ignore":
        encoder = OrdinalEncoder(
            categories=[list(spec.order or [])],
            handle_unknown="use_encoded_value",
            unknown_value=np.nan,
        )
        return Pipeline(
            [("encode", encoder), ("impute", SimpleImputer(strategy="most_frequent"))]
        )
    return OrdinalEncoder(categories=[list(spec.order or [])], handle_unknown="error")


def build_preprocessor(cfg: AppConfig) -> ColumnTransformer:
    """Unfitted preprocessing transformer; output is a pandas DataFrame."""
    cat_cfg = cfg.preprocessing.categorical
    transformers = [(NUMERIC_BLOCK, _numeric_block(cfg), cfg.numeric_features)]
    for column in cfg.categorical_features:
        block = _categorical_block(cat_cfg.encoders[column], cat_cfg.handle_unknown)
        transformers.append((f"cat_{column}", block, [column]))
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=True,
    )
    preprocessor.set_output(transform="pandas")
    return preprocessor
