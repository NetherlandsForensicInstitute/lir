from pathlib import Path

import numpy as np
import pytest

from lir import FeatureData, InstanceData, registry
from lir.config.base import ConfigValue, GenericConfigParser
from lir.config.data import data_provider, parse_data_setup
from lir.data_setup.train_test_split import SplitDataSetup
from lir.splitting_strategies import FullTrainTestSplit


@pytest.mark.parametrize(
    'instances',
    [
        pytest.param(FeatureData(features=np.arange(8).reshape((-1, 2))), id='unlabeled'),
        pytest.param(
            FeatureData(features=np.arange(8).reshape((-1, 2)), hypothesis=np.array([0, 1, 0, 1])),
            id='labeled',
        ),
        pytest.param(
            FeatureData(features=np.arange(8).reshape((-1, 2)), source_ids=np.array([0, 1, 1, 2])),
            id='sources',
        ),
        pytest.param(
            FeatureData(
                features=np.arange(8).reshape((-1, 2)),
                source_ids=np.array([[0, 1], [0, 2], [1, 2], [3, 4]]),
            ),
            id='pairs',
        ),
    ],
)
def test_full_train_test_split(instances: InstanceData):
    splits = list(FullTrainTestSplit().apply(instances))

    assert len(splits) == 1
    training, testing = splits[0]
    assert len(training) == len(instances)
    assert len(testing) == len(instances)
    assert np.array_equal(training.features, instances.features)
    assert np.array_equal(testing.features, instances.features)


def test_registry():
    strategy = registry.get('splitting_strategies.train_test_full', default_config_parser=GenericConfigParser)
    assert isinstance(strategy, GenericConfigParser)
    assert strategy.component_class is FullTrainTestSplit


@data_provider
def load_data() -> FeatureData:
    return FeatureData(features=np.arange(4).reshape((-1, 1)))


def test_parse_data_setup():
    config = ConfigValue.wrap(
        ['data'],
        {
            'setup': 'splits',
            'provider': {'method': 'tests.splitting_strategies.test_full.load_data'},
            'splits': {'strategy': 'train_test_full'},
        },
    )
    setup = parse_data_setup(config, Path('/'))

    assert isinstance(setup, SplitDataSetup)
    assert isinstance(setup.strategy, FullTrainTestSplit)

    splits = setup.get_train_inference_pairs()
    assert len(splits) == 1
    training, testing = splits[0]
    assert len(training) == len(testing) == 4
