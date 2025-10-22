import numpy as np

from lir.data.data_strategies import MulticlassTrainTestSplit
from lir.data.datasets.synthesized_normal_multiclass import SynthesizedDimension, \
    SynthesizedNormalMulticlassData


def test_multiclass_train_test_split():
    dimensions = [SynthesizedDimension(0, 1, 0.2), SynthesizedDimension(0, 1, 0.2)]
    data = SynthesizedNormalMulticlassData(population_size=100, sources_size=3, seed=0, dimensions=dimensions)
    strategy = MulticlassTrainTestSplit(data, test_size=0.5, seed=0)
    features, labels, meta = data.get_instances()
    for data_train, data_test in strategy:
        labels_train = data_train[1]
        labels_test = data_test[1]
        assert len(np.unique(labels_train)) + len(np.unique(labels_test)) == len(np.unique(labels))
        assert len(labels_train) + len(labels_test) == len(labels)


def test_multiclass_train_test_split_seed():
    dimensions = [SynthesizedDimension(0, 1, 0.2), SynthesizedDimension(0, 1, 0.2)]
    data = SynthesizedNormalMulticlassData(population_size=100, sources_size=3, seed=0, dimensions=dimensions)

    ref_strategy = MulticlassTrainTestSplit(data, test_size=0.5, seed=0)
    ref_train, ref_test = next(iter(ref_strategy))

    # test:
    # - another MulticlassTrainTestSplit instance with the same seed should yield the same results
    # - the same instance should yield the same results twice
    strategies = [MulticlassTrainTestSplit(data, test_size=0.5, seed=0)] * 2
    for strategy in strategies:
        for data_train, data_test in strategy:
            assert np.all(data_train[0] == ref_train[0])
            assert np.all(data_test[0] == ref_test[0])

    # test:
    # - another MulticlassTrainTestSplit instance with another seed should yield different results
    strategy = MulticlassTrainTestSplit(data, test_size=0.5, seed=1)
    for data_train, data_test in strategy:
        assert not np.all(data_train[0] == ref_train[0])
        assert not np.all(data_test[0] == ref_test[0])
