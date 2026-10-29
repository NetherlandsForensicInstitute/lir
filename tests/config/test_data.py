from pathlib import Path

import numpy as np
import pytest

from lir import FeatureData, registry
from lir.config.base import ConfigParser, ConfigValue
from lir.config.data import data_provider, parse_data_setup
from lir.data_setup.train_test_split import SplitDataSetup


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
        (np.arange(4), load_data.parse(ConfigValue.wrap([], {}), Path('/')).get_instances()),
        (np.array([1, 2]), load_data_with_args(first=1, second=2)),
        (
            np.array([1, 2]),
            load_data_with_args.parse(ConfigValue.wrap([], {'first': 1, 'second': 2}), Path('/')).get_instances(),
        ),
    ],
)
def test_load_data(expected_result: np.ndarray, dataset: FeatureData):
    assert np.all(expected_result == dataset.features.reshape(-1))


def test_parse_data_setup_without_setup_key_warns_deprecation():
    config = ConfigValue.wrap(
        ['data'],
        {
            'provider': {
                'method': 'synthesized_normal_binary',
                'seed': 42,
                'h1': {'mean': 0, 'std': 1, 'size': 10},
                'h2': {'mean': 2, 'std': 1, 'size': 10},
            },
            'splits': {'strategy': 'train_test_instances', 'test_size': 0.5, 'seed': 1},
        },
    )
    with pytest.warns(DeprecationWarning, match='data: missing key: setup'):
        setup = parse_data_setup(config, Path('/'))
    assert isinstance(setup, SplitDataSetup)
