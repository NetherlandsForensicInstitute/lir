import numpy as np
import pytest

from lir import FeatureData
from lir.transform.data_enforcer import DataEnforcer


def test_data_enforcer():
    features = np.array([[1, 2.0], [3, 4.0], [5, 6.0]], dtype=np.float16)
    instances = FeatureData(features=features)

    data_enforcer = DataEnforcer()
    data_enforcer.fit(instances)

    # Apply the DataEnforcer to the same data (should pass)
    transformed_instances = data_enforcer.apply(instances)
    assert np.array_equal(transformed_instances.features, features)

    # Apply a different set of features, but with the same data types (should pass)
    features_different = np.array([[7, 8.0]], dtype=np.float16)
    instances_different = FeatureData(features=features_different)
    transformed_instances_different = data_enforcer.apply(instances_different)
    assert np.array_equal(transformed_instances_different.features, features_different)

    # Invalid type: float64 cannot be cast to float16 without loss of precision.
    features_invalid = np.array([[1.5, 2], [3.5, 4], [5.5, 6]], dtype=np.float64)
    instances_invalid = FeatureData(features=features_invalid)
    pytest.raises(TypeError, data_enforcer.apply, instances_invalid)

    # Invalid number of features
    features_invalid = np.array([[1, 2.0, 3], [3, 4.0, 5], [5, 6.0, 7]], dtype=np.float16)
    instances_invalid = FeatureData(features=features_invalid)
    pytest.raises(ValueError, data_enforcer.apply, instances_invalid)
