import numpy as np

from lir.data.models import FeatureData
from lir.data_strategies import PredefinedTrainTestSplit


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
