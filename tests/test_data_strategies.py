import numpy as np

from lir.data.data_strategies import MulticlassTrainTestSplit, PredefinedTrainTestSplit
from lir.data.datasets.synthesized_normal_multiclass import (
    SynthesizedDimension,
    SynthesizedNormalMulticlassData,
)
from lir.data.models import FeatureData, DataSet, InstanceData


def test_multiclass_train_test_split():
    dimensions = [SynthesizedDimension(0, 1, 0.2), SynthesizedDimension(0, 1, 0.2)]
    data = SynthesizedNormalMulticlassData(
        population_size=100, sources_size=3, seed=0, dimensions=dimensions
    )
    strategy = MulticlassTrainTestSplit(data, test_size=0.5, seed=0)
    instances = data.get_instances()
    for data_train, data_test in strategy:
        assert len(np.unique(data_train.source_ids)) + len(np.unique(data_test.source_ids)) == len(
            np.unique(instances.source_ids)
        )
        assert len(data_train.source_ids) + len(data_test.source_ids) == len(instances.source_ids)


def test_multiclass_train_test_split_seed():
    dimensions = [SynthesizedDimension(0, 1, 0.2), SynthesizedDimension(0, 1, 0.2)]
    data = SynthesizedNormalMulticlassData(
        population_size=100, sources_size=3, seed=0, dimensions=dimensions
    )

    ref_strategy = MulticlassTrainTestSplit(data, test_size=0.5, seed=0)
    ref_train, ref_test = next(iter(ref_strategy))

    # test:
    # - another MulticlassTrainTestSplit instance with the same seed should yield the same results
    # - the same instance should yield the same results twice
    strategies = [MulticlassTrainTestSplit(data, test_size=0.5, seed=0)] * 2
    for strategy in strategies:
        for data_train, data_test in strategy:
            assert data_train == ref_train
            assert data_test == ref_test

    # test:
    # - another MulticlassTrainTestSplit instance with another seed should yield different results
    strategy = MulticlassTrainTestSplit(data, test_size=0.5, seed=1)
    for data_train, data_test in strategy:
        assert not data_train == ref_train
        assert not data_test == ref_test


class _MemoryDataProvider(DataSet):
    def __init__(self, instances: InstanceData):
        self.instances = instances

    def get_instances(self) -> InstanceData:
        return self.instances


def test_predefined_role_splitter():
    instances = FeatureData(features=np.arange(16).reshape(-1, 2), role_assignments=np.array(["train", "test"]).repeat(4))
    splitter = PredefinedTrainTestSplit(_MemoryDataProvider(instances))
    splits = list(splitter)
    assert len(splits) == 1, "exactly one split"

    training_set, test_set = splits[0]
    expected_training_features = np.arange(8).reshape(-1, 2)
    expected_test_features = expected_training_features + 8
    assert training_set == FeatureData(features=expected_training_features, role_assignments=np.array(["train"] * 4))
    assert test_set == FeatureData(features=expected_test_features, role_assignments=np.array(["test"] * 4))

