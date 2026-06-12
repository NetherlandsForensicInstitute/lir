import numpy as np

from lir.data.models import FeatureData
from lir.data_strategies import PairsTrainTestSplit
from lir.transform.pairing import InstancePairing


def test_paired_train_test_split():
    instances = FeatureData(features=np.ones(20), source_ids=np.arange(10).repeat(2))
    pairs = InstancePairing().pair(instances)
    training_pairs, test_pairs = next(iter(PairsTrainTestSplit(test_size=0.5, seed=0).apply(pairs)))
    assert len(training_pairs) == len(test_pairs) == 10 * 9 / 2
    assert len(np.unique(training_pairs.source_ids)) == 5, 'half of the sources are for training'
    assert len(np.unique(test_pairs.source_ids)) == 5, 'half of the sources are for testing'
    assert len(np.unique(np.concatenate([test_pairs.source_ids, training_pairs.source_ids]))) == 10, (
        'all sources are for training or testing'
    )
