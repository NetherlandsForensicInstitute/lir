from pathlib import Path

import numpy as np
import pytest

from lir import FeatureData, registry
from lir.config.base import ConfigParser, ContextAwareDict
from lir.config.data import data_provider


@data_provider
def load_data() -> FeatureData:
    return FeatureData(features=np.arange(4).reshape((-1, 1)))


@data_provider
def load_data_with_args(first: int, second: int) -> FeatureData:
    return FeatureData(features=np.array([[first], [second]]))


def test_registry():
    assert isinstance(load_data, ConfigParser)
    assert load_data == registry.get('tests.config.test_data.load_data')


@pytest.mark.parametrize(
    'expected_result,dataset',
    [
        (np.arange(4), load_data()),
        (np.arange(4), load_data.parse(ContextAwareDict([]), Path('/')).get_instances()),
        (np.array([1, 2]), load_data_with_args(first=1, second=2)),
        (
            np.array([1, 2]),
            load_data_with_args.parse(ContextAwareDict([], {'first': 1, 'second': 2}), Path('/')).get_instances(),
        ),
    ],
)
def test_load_data(expected_result: np.ndarray, dataset: FeatureData):
    assert np.all(expected_result == dataset.features.reshape(-1))
