import numpy as np
import pytest

from lir import DataStrategy
from lir.data_strategies import SourcesCrossValidation, SourcesTrainTestSplit
from lir.data_strategies.sources import LeaveOneSourceOut
from lir.datasets.synthesized_normal_multiclass import (
    SynthesizedDimension,
    SynthesizedNormalMulticlassData,
)


def test_multiclass_train_test_split():
    dimensions = [SynthesizedDimension(0, 1, 0.2), SynthesizedDimension(0, 1, 0.2)]
    data = SynthesizedNormalMulticlassData(population_size=100, sources_size=3, seed=0, dimensions=dimensions)
    strategy = SourcesTrainTestSplit(test_size=0.5, seed=0)
    instances = data.get_instances()
    for data_train, data_test in strategy.apply(instances):
        assert len(np.unique(data_train.source_ids)) + len(np.unique(data_test.source_ids)) == len(
            np.unique(instances.source_ids)
        )
        assert len(data_train.source_ids) + len(data_test.source_ids) == len(instances.source_ids)


def test_multiclass_train_test_split_seed():
    dimensions = [SynthesizedDimension(0, 1, 0.2), SynthesizedDimension(0, 1, 0.2)]
    data = SynthesizedNormalMulticlassData(
        population_size=100, sources_size=3, seed=0, dimensions=dimensions
    ).get_instances()

    ref_strategy = SourcesTrainTestSplit(test_size=0.5, seed=0)
    ref_train, ref_test = next(iter(ref_strategy.apply(data)))

    # test:
    # - another MulticlassTrainTestSplit instance with the same seed should yield the same results
    # - the same instance should yield the same results twice
    strategies = [SourcesTrainTestSplit(test_size=0.5, seed=0)] * 2
    for strategy in strategies:
        for data_train, data_test in strategy.apply(data):
            assert data_train == ref_train
            assert data_test == ref_test

    # test:
    # - another MulticlassTrainTestSplit instance with another seed should yield different results
    strategy = SourcesTrainTestSplit(test_size=0.5, seed=1)
    for data_train, data_test in strategy.apply(data):
        assert data_train != ref_train
        assert data_test != ref_test


@pytest.mark.parametrize(
    'strategy',
    [
        SourcesTrainTestSplit(test_size=0.5, seed=0),
        SourcesCrossValidation(folds=20),  # no shuffle
        SourcesCrossValidation(folds=20, random_state=0),
        LeaveOneSourceOut(),
    ],
)
def test_reproducibility(strategy: DataStrategy):
    dimensions = [SynthesizedDimension(0, 1, 0.2), SynthesizedDimension(0, 1, 0.2)]
    data = SynthesizedNormalMulticlassData(population_size=100, sources_size=3, seed=0, dimensions=dimensions)
    instances = data.get_instances()
    splits1 = list(strategy.apply(instances))
    splits2 = list(strategy.apply(instances))
    for i in range(len(splits1)):
        assert np.all(splits1[i][0].features == splits2[i][0].features)


def test_leave_one_out():
    dimensions = [SynthesizedDimension(0, 1, 0.2), SynthesizedDimension(0, 1, 0.2)]
    data = SynthesizedNormalMulticlassData(population_size=100, sources_size=3, seed=0, dimensions=dimensions)
    instances = data.get_instances()
    strategy = LeaveOneSourceOut()
    assert len(list(strategy.apply(instances))) == 100
