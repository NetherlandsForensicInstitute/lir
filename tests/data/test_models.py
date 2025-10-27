from typing import Any, Self

import numpy as np
import pytest
from pydantic import ValidationError

from lir.data.models import InstanceData, FeatureData, LLRData


class _BareInstanceData(InstanceData):
    def replace(self, **kwargs: Any) -> Self:
        return self


def test_instance_data():
    _BareInstanceData(labels=None)
    _BareInstanceData(labels=np.zeros((10,)))
    _BareInstanceData(labels=np.zeros((10,)), meta=np.zeros((10,)))  # type: ignore
    _BareInstanceData(labels=np.ones((10,)))
    _BareInstanceData(labels=np.concatenate([np.zeros((10,)), np.ones((10,))]))

    assert {"labels"} == set(_BareInstanceData(labels=None).all_fields)
    assert {"labels", "meta"} == set(_BareInstanceData(labels=None, meta=1).all_fields)

    # illegal labels type
    with pytest.raises(ValidationError):
        _BareInstanceData(labels=1)  # type: ignore

    # illegal label dimensions
    with pytest.raises(ValidationError):
        _BareInstanceData(labels=np.ones((10, 1)))

    # illegal operation
    with pytest.raises(ValidationError):
        instances = _BareInstanceData(labels=np.array([0, 1]))
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
