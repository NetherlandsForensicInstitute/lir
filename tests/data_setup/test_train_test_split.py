from pathlib import Path

import numpy as np
import pytest

from lir.config import ConfigValue, YamlParseError
from lir.config.data import parse_data_setup
from lir.data.models import DataProvider, FeatureData, InstanceData
from lir.data_setup.train_test_split import SplitDataSetup
from lir.splitting_strategies.labels import TrainTestSplit


class CountingProvider(DataProvider):
    """Data provider that counts how often `get_instances()` is called."""

    def __init__(self):
        self.call_count = 0

    def get_instances(self) -> InstanceData:
        self.call_count += 1
        labels = np.zeros(20, dtype=int)
        labels[::2] = 1
        return FeatureData(features=np.arange(20).reshape((-1, 1)), hypothesis=labels)


def _data_section(**extra) -> dict:
    return {
        'setup': 'splits',
        'provider': {
            'method': 'synthesized_normal_binary',
            'seed': 42,
            'h1': {'mean': 0, 'std': 1, 'size': 10},
            'h2': {'mean': 2, 'std': 1, 'size': 10},
        },
        'splits': {'strategy': 'train_test_instances', 'test_size': 0.5, 'seed': 1},
        **extra,
    }


def test_parse_split_data_unknown_key_raises():
    config = ConfigValue.wrap(['data'], _data_section(filterr={'method': 'identity'}))
    with pytest.raises(YamlParseError, match='data: unrecognized argument: filterr'):
        parse_data_setup(config, Path())


def _make_setup(provider: DataProvider, use_cache: bool | None = None) -> SplitDataSetup:
    if use_cache is None:
        return SplitDataSetup(provider, TrainTestSplit(test_size=0.5, seed=1), None)
    return SplitDataSetup(provider, TrainTestSplit(test_size=0.5, seed=1), None, use_cache=use_cache)


def test_default_use_cache_caches_train_inference_pairs():
    provider = CountingProvider()
    setup = _make_setup(provider)

    next(iter(setup.get_train_inference_pairs()))
    next(iter(setup.get_train_inference_pairs()))

    assert provider.call_count == 1


def test_disabled_use_cache_does_not_cache_train_inference_pairs():
    provider = CountingProvider()
    setup = _make_setup(provider, use_cache=False)

    next(iter(setup.get_train_inference_pairs()))
    next(iter(setup.get_train_inference_pairs()))

    assert provider.call_count == 2
