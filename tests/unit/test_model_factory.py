import pytest
from sklearn.pipeline import Pipeline

from mlops_pipeline.config.schema import ModelSpec
from mlops_pipeline.models import build_estimator, build_pipeline
from mlops_pipeline.models.factory import ESTIMATORS, MODEL_STEP, PREPROCESSING_STEP


def test_pipeline_is_preprocessing_then_model(settings):
    pipe = build_pipeline(settings.config, settings.config.model.baseline)
    assert isinstance(pipe, Pipeline)
    assert [n for n, _ in pipe.steps] == [PREPROCESSING_STEP, MODEL_STEP]
    assert not hasattr(pipe.named_steps[MODEL_STEP], "coef_")  # unfitted


def test_hyperparameters_come_from_config(settings):
    cand = settings.config.model.candidate
    est = build_estimator(cand, seed=1)
    for k, v in cand.hyperparameters.items():
        assert est.get_params()[k] == v


@pytest.mark.parametrize("algorithm", sorted(ESTIMATORS))
def test_seed_injected_only_where_supported(algorithm):
    est = build_estimator(ModelSpec(name="m", algorithm=algorithm, hyperparameters={}), seed=7)
    params = est.get_params()
    if "random_state" in params:
        assert params["random_state"] == 7


def test_invalid_hyperparameter_gives_clear_error():
    spec = ModelSpec(name="m", algorithm="linear_regression", hyperparameters={"nonsense": 1})
    with pytest.raises(ValueError, match="invalid hyperparameters"):
        build_estimator(spec, seed=0)
