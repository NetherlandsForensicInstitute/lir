from typing import Any, cast

import numpy as np

from lir.algorithms.mcmc import McmcLLRModel, McmcModel
from lir.bounding import LLRBounder
from lir.data.models import FeatureData, LLRData


def _build_model(distribution: str) -> McmcModel:
    model = McmcModel(distribution=distribution, parameters={'mu': {'prior': 'normal', 'mu': 0, 'sigma': 1}})
    model.parameter_samples = {
        'mu': np.array([0.0, 0.5, 1.0]),
        'sigma': np.array([1.0, 1.0, 1.0]),
    }
    return model


def test_mcmc_transform_accepts_normal_distribution_name():
    model = _build_model('normal')
    features = np.array([[0.0], [1.0]])

    logp = model.transform(features)

    assert logp.shape == (2, 3)


def test_mcmc_transform_norm_alias_matches_normal():
    features = np.array([[0.0], [1.0]])
    normal_model = _build_model('normal')
    norm_model = _build_model('norm')

    logp_normal = normal_model.transform(features)
    logp_norm = norm_model.transform(features)

    np.testing.assert_allclose(logp_normal, logp_norm)


class _NoopBounder(LLRBounder):
    def calculate_bounds(self, llrdata: LLRData) -> tuple[None, None]:
        return None, None


class _DummyModel:
    def __init__(self, output: np.ndarray):
        self.output = output

    def fit(self, features: np.ndarray):
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        return self.output


def test_mcmc_apply_with_bounding_handles_llrdata_correctly():
    instances = FeatureData(features=np.array([[0.0], [1.0]]), labels=np.array([0, 1]))

    llr_model = McmcLLRModel(
        distribution_h1='normal',
        parameters_h1={'mu': {'prior': 'normal', 'mu': 0, 'sigma': 1}},
        distribution_h2='normal',
        parameters_h2={'mu': {'prior': 'normal', 'mu': 0, 'sigma': 1}},
        bounding=_NoopBounder(),
    )
    cast(Any, llr_model).model_h1 = _DummyModel(output=np.array([[0.2, 0.4], [0.1, 0.3]]))
    cast(Any, llr_model).model_h2 = _DummyModel(output=np.array([[0.1, 0.2], [0.0, 0.1]]))

    llr_model.fit(instances)
    result = llr_model.apply(instances)

    assert result.features.shape == (2, 3)
