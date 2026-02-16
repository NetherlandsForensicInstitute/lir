import numpy as np

from lir.data.data_strategies import MulticlassTrainTestSplit, PairedInstancesTrainTestSplit, PredefinedTrainTestSplit
from lir.data.models import FeatureData
from lir.datasets.synthesized_normal_multiclass import SynthesizedDimension, SynthesizedNormalMulticlassData
from lir.transform.pairing import InstancePairing


def test_multiclass_train_test_split():
    dimensions = [SynthesizedDimension(0, 1, 0.2), SynthesizedDimension(0, 1, 0.2)]
    data = SynthesizedNormalMulticlassData(population_size=100, sources_size=3, seed=0, dimensions=dimensions)
    strategy = MulticlassTrainTestSplit(test_size=0.5, seed=0)
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

    ref_strategy = MulticlassTrainTestSplit(test_size=0.5, seed=0)
    ref_train, ref_test = next(iter(ref_strategy.apply(data)))

    # test:
    # - another MulticlassTrainTestSplit instance with the same seed should yield the same results
    # - the same instance should yield the same results twice
    strategies = [MulticlassTrainTestSplit(test_size=0.5, seed=0)] * 2
    for strategy in strategies:
        for data_train, data_test in strategy.apply(data):
            assert data_train == ref_train
            assert data_test == ref_test

    # test:
    # - another MulticlassTrainTestSplit instance with another seed should yield different results
    strategy = MulticlassTrainTestSplit(test_size=0.5, seed=1)
    for data_train, data_test in strategy.apply(data):
        assert data_train != ref_train
        assert data_test != ref_test


def test_paired_train_test_split():
    instances = FeatureData(features=np.ones(20), source_ids=np.arange(10).repeat(2))
    pairs = InstancePairing().pair(instances)
    training_pairs, test_pairs = next(iter(PairedInstancesTrainTestSplit(test_size=0.5, seed=0).apply(pairs)))
    assert len(training_pairs) == len(test_pairs) == 10 * 9 / 2
    assert len(np.unique(training_pairs.source_ids)) == 5, 'half of the sources are for training'
    assert len(np.unique(test_pairs.source_ids)) == 5, 'half of the sources are for testing'
    assert len(np.unique(np.concatenate([test_pairs.source_ids, training_pairs.source_ids]))) == 10, (
        'all sources are for training or testing'
    )


def test_predefined_role_splitter():
    instances = FeatureData(
        features=np.arange(16).reshape(-1, 2), role_assignments=np.array(['train', 'test']).repeat(4)
    )
    splitter = PredefinedTrainTestSplit()
    splits = list(splitter.apply(instances))
    assert len(splits) == 1, 'exactly one split'

    training_set, test_set = splits[0]
    expected_training_features = np.arange(8).reshape(-1, 2)
    expected_test_features = expected_training_features + 8
    assert training_set == FeatureData(features=expected_training_features, role_assignments=np.array(['train'] * 4))
    assert test_set == FeatureData(features=expected_test_features, role_assignments=np.array(['test'] * 4))
