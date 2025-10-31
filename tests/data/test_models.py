from typing import Any, Self

import numpy as np
import pytest
from pydantic import ValidationError

from lir.data.models import InstanceData, FeatureData, LLRData, concatenate_instances


class BareData(InstanceData):
    def __len__(self) -> int:
        if self.labels is None:
            raise ValueError()
        else:
            return self.labels.shape[0]


def test_instance_data():
    BareData(labels=None)
    BareData(labels=np.zeros((10,)))
    BareData(labels=np.zeros((10,)), meta=np.zeros((10,)))  # type: ignore
    BareData(labels=np.ones((10,)))
    BareData(labels=np.concatenate([np.zeros((10,)), np.ones((10,))]))

    # test all_fields property
    assert {"labels"} == set(BareData(labels=None).all_fields)
    assert {"labels", "meta"} == set(BareData(labels=None, meta=1).all_fields)

    # test __eq__ method
    assert BareData(labels=np.ones((10,))) == BareData(labels=np.ones((10,)))
    assert not (BareData(labels=np.zeros((10,))) == BareData(labels=np.ones((10,))))
    assert BareData(labels=np.zeros((10,))) != BareData(labels=np.ones((10,)))
    assert not (BareData(labels=np.zeros((10,))) == BareData(labels=np.ones((10,))))
    assert BareData(labels=np.ones((10,)), meta=3) == BareData(labels=np.ones((10,)), meta=3)
    assert BareData(labels=np.ones((10,)), meta=2) != BareData(labels=np.ones((10,)), meta=3)
    assert BareData(labels=np.ones((10,)), meta=2) != BareData(labels=np.ones((10,)))

    # test slicing
    assert np.all(BareData(labels=np.arange(10))[:5].labels == np.arange(5))
    assert BareData(labels=np.arange(10))[8:9].labels == np.array([8])

    # illegal labels type
    with pytest.raises(ValidationError):
        BareData(labels=1)  # type: ignore

    # illegal label dimensions
    with pytest.raises(ValidationError):
        BareData(labels=np.ones((10, 1)))

    # illegal operation
    with pytest.raises(ValidationError):
        instances = BareData(labels=np.array([0, 1]))
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


def test_concatenate():
    data = FeatureData(features=np.ones((10, 2)))
    assert concatenate_instances(data, data) == FeatureData(features=np.ones((20, 2)))

    data = FeatureData(features=np.ones((10, 2)), extra1=3, extra2=None)
    assert concatenate_instances(data, data) == FeatureData(features=np.ones((20, 2)), extra1=3, extra2=None)

    data = FeatureData(features=np.ones((10, 2)), extra1=[1, 2])
    assert concatenate_instances(data, data) == FeatureData(features=np.ones((20, 2)), extra1=[1, 2])

    with pytest.raises(ValueError):
        concatenate_instances(FeatureData(features=np.ones((10, 2)), extra1=3), FeatureData(features=np.ones((10, 2)), extra1=4))

    with pytest.raises(ValueError):
        concatenate_instances(FeatureData(features=np.ones((10, 2)), extra1=[1, 2]), FeatureData(features=np.ones((10, 2)), extra1=[2, 1]))


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
