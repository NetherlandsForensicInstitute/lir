from pathlib import Path

import numpy as np
import pytest

from lir.config import ConfigValue
from lir.config.data import parse_data_setup
from lir.data.models import DataProvider, FeatureData, InstanceData
from lir.data_setup.inference import InferenceDataSetup


class CountingProvider(DataProvider):
    """Data provider that counts how often `get_instances()` is called."""

    def __init__(self):
        self.call_count = 0

    def get_instances(self) -> InstanceData:
        self.call_count += 1
        return FeatureData(features=np.arange(10).reshape((-1, 1)), hypothesis=np.zeros(10, dtype=int))


def _inference_data_section(**extra) -> dict:
    return {
        'setup': 'inference',
        'training': {
            'method': 'synthesized_normal_binary',
            'seed': 42,
            'h1': {'mean': 0, 'std': 1, 'size': 10},
            'h2': {'mean': 2, 'std': 1, 'size': 10},
        },
        'inference': {
            'method': 'synthesized_normal_binary',
            'seed': 43,
            'h1': {'mean': 0, 'std': 1, 'size': 5},
            'h2': {'mean': 2, 'std': 1, 'size': 5},
        },
        **extra,
    }


def test_inference_data_setup(synthesized_normal_data_provider: DataProvider):
    data = InferenceDataSetup(synthesized_normal_data_provider, synthesized_normal_data_provider)
    assert len(list(data.get_train_inference_pairs())) == 1
    assert len(list(data.get_train_inference_pairs())) == 1
    pair = next(iter(data.get_train_inference_pairs()))
    assert pair[0] is not None
    assert pair[1] is not None
    assert len(pair[0]) == len(synthesized_normal_data_provider.get_instances())
    assert len(pair[1]) == len(synthesized_normal_data_provider.get_instances())

    data = InferenceDataSetup(None, synthesized_normal_data_provider)
    pair = next(iter(data.get_train_inference_pairs()))
    assert pair[0] is None
    assert pair[1] is not None

    data = InferenceDataSetup(synthesized_normal_data_provider, None)
    pair = next(iter(data.get_train_inference_pairs()))
    assert pair[0] is not None
    assert pair[1] is None


def test_inference_data_setup_requires_a_provider():
    with pytest.raises(ValueError, match='specify at least one of training data and inference data'):
        InferenceDataSetup(None, None)


def test_inference_data_setup_caches_instances():
    training_provider = CountingProvider()
    inference_provider = CountingProvider()
    setup = InferenceDataSetup(training_provider, inference_provider)

    next(iter(setup.get_train_inference_pairs()))
    next(iter(setup.get_train_inference_pairs()))

    assert training_provider.call_count == 1
    assert inference_provider.call_count == 1


def test_parse_inference_data_setup():
    config = ConfigValue.wrap(['data'], _inference_data_section())
    setup = parse_data_setup(config, Path())
    assert isinstance(setup, InferenceDataSetup)
    pair = next(iter(setup.get_train_inference_pairs()))
    assert pair[0] is not None
    assert pair[1] is not None
    assert len(pair[0]) == 20
    assert len(pair[1]) == 10


def test_parse_inference_data_setup_empty_raises():
    config = ConfigValue.wrap(['data'], {'setup': 'inference'})
    with pytest.raises(ValueError, match='specify at least one of training data and inference data'):
        parse_data_setup(config, Path())
