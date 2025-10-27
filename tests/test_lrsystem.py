import numpy as np
import pytest
from pydantic import ValidationError

from lir.lrsystems.lrsystems import InstanceData, FeatureData, LLRData


def test_instance_data():
    InstanceData(labels=None)
    InstanceData(labels=np.zeros((10,)))
    InstanceData(labels=np.zeros((10,)), meta=np.zeros((10,)))  # type: ignore
    InstanceData(labels=np.ones((10,)))
    InstanceData(labels=np.concatenate([np.zeros((10,)), np.ones((10,))]))

    # illegal labels type
    with pytest.raises(ValidationError):
        InstanceData(labels=1)  # type: ignore

    # illegal label dimensions
    with pytest.raises(ValidationError):
        InstanceData(labels=np.ones((10, 1)))

    # illegal label values
    with pytest.raises(ValidationError):
        InstanceData(labels=np.ones((10,)) * 2)

    # illegal label values
    with pytest.raises(ValidationError):
        InstanceData(labels=np.array([0, 1, np.nan]))

    # illegal operation
    with pytest.raises(ValidationError):
        instances = InstanceData(labels=np.array([0, 1]))
        instances.labels = np.array([1, 1])


def test_feature_data():
    FeatureData(features=np.ones((10, 2)), labels=None)
    FeatureData(features=np.ones((10, 2)), labels=np.ones((10,)))
    with pytest.raises(ValidationError):
        FeatureData(features=np.ones((10, 2)), labels=np.ones((11,)))

    # illegal operation
    with pytest.raises(ValidationError):
        instances = FeatureData(features=np.ones((10, 2)), labels=np.ones((10,)))
        instances.features = np.ones((10, 2))


def test_llr_data():
    LLRData(features=np.ones((10,)))
    LLRData(features=np.ones((10, 1)))
    LLRData(features=np.ones((10, 3)))
    LLRData(features=np.ones((10,)), labels=np.ones(10))

    with pytest.raises(ValidationError):
        LLRData(features=np.ones((10, 2)))

    with pytest.raises(ValidationError):
        LLRData(features=np.ones((10, 4)))

    with pytest.raises(ValidationError):
        LLRData(features=np.ones((10, 3, 1)))

    with pytest.raises(ValidationError):
        LLRData(features=np.ones((10,)), labels=np.ones(11))

    llr_values = np.arange(30).reshape(10, 3)
    assert np.all(LLRData(features=llr_values).llrs == llr_values[:, 0])
    assert np.all(LLRData(features=llr_values[:, 0]).llrs == llr_values[:, 0])
    assert np.all(LLRData(features=llr_values[:, 0:1]).llrs == llr_values[:, 0])
    assert np.all(LLRData(features=llr_values).llr_intervals == llr_values[:, 1:3])
