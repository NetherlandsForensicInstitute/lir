import numpy as np
import pytest

from lir import FeatureData, InstanceData, registry
from lir.config.base import GenericConfigParser
from lir.data_strategies import LeaveOneCategoryOut


@pytest.mark.parametrize(
    'instances,n_splits,expected_training_sizes,expected_test_sizes',
    [
        (
            FeatureData(
                features=np.zeros((3, 2)),
                my_category=np.array(['a', 'b', 'c']),
            ),
            3,
            [2, 2, 2],
            [1, 1, 1],
        ),
        (
            FeatureData(
                features=np.zeros((3, 2)),
                my_category=np.array(['a', 'a', 'b']),
            ),
            2,
            [1, 2],
            [2, 1],
        ),
        (
            FeatureData(
                features=np.zeros((100, 2)),
                hypothesis=np.stack([np.zeros(50), np.ones(50)], axis=1).reshape(-1),
                my_category=np.repeat(np.arange(5), 20),
            ),
            5,
            np.ones(5) * 80,
            np.ones(5) * 20,
        ),
    ],
)
def test_leave_one_category_out(
    instances: InstanceData, n_splits: int, expected_training_sizes: int, expected_test_sizes: int
):
    strategy = LeaveOneCategoryOut('my_category')
    splits = list(strategy.apply(instances))
    assert len(splits) == n_splits
    assert np.all(np.array([len(splits[i][0]) for i in range(n_splits)]) == np.array(expected_training_sizes))
    assert np.all(np.array([len(splits[i][1]) for i in range(n_splits)]) == np.array(expected_test_sizes))


def test_registry():
    assert registry.get('data_strategies.leave_one_category_out', default_config_parser=GenericConfigParser)
