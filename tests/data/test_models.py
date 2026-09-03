from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from lir.data.models import FeatureData, InstanceData, LLRData, PairedFeatureData, concatenate_instances


class BareData(InstanceData):
    def __len__(self) -> int:
        if self.hypothesis is None:
            raise ValueError()
        else:
            return self.hypothesis.shape[0]


def test_instance_data():
    BareData(hypothesis=None)
    BareData(hypothesis=np.zeros((10,)))
    BareData(hypothesis=np.zeros((10,)), meta=np.zeros((10,)))  # type: ignore
    BareData(hypothesis=np.ones((10,)))
    BareData(hypothesis=np.concatenate([np.zeros((10,)), np.ones((10,))]))

    # test all_fields property
    assert {'hypothesis', 'source_ids'} == set(BareData(hypothesis=None).all_fields)
    assert {'hypothesis', 'source_ids', 'meta'} == set(BareData(hypothesis=None, meta=1).all_fields)

    # test __eq__ method
    assert BareData(hypothesis=np.ones((10,))) == BareData(hypothesis=np.ones((10,)))
    assert BareData(hypothesis=np.zeros((10,))) != BareData(hypothesis=np.ones((10,)))
    assert BareData(hypothesis=np.zeros((10,))) != BareData(hypothesis=np.ones((10,)))
    assert BareData(hypothesis=np.zeros((10,))) != BareData(hypothesis=np.ones((10,)))
    assert BareData(hypothesis=np.ones((10,)), meta=3) == BareData(hypothesis=np.ones((10,)), meta=3)
    assert BareData(hypothesis=np.ones((10,)), meta=2) != BareData(hypothesis=np.ones((10,)), meta=3)
    assert BareData(hypothesis=np.ones((10,)), meta=2) != BareData(hypothesis=np.ones((10,)))

    # test slicing
    assert np.all(BareData(source_ids=np.arange(10))[:5].source_ids_1d == np.arange(5))
    assert np.all(BareData(hypothesis=np.repeat([0, 1, 0, 1], 2))[:5].hypothesis == np.array([0, 0, 1, 1, 0]))
    assert BareData(source_ids=np.arange(10))[8:9].source_ids_1d == np.array([8])

    # illegal labels type
    with pytest.raises(ValidationError):
        BareData(hypothesis=1)  # type: ignore

    # illegal label dimensions
    with pytest.raises(ValidationError):
        BareData(hypothesis=np.ones((10, 1)))

    # illegal operation
    with pytest.raises(ValidationError):
        instances = BareData(hypothesis=np.array([0, 1]))
        instances.hypothesis = np.array([1, 1])


def test_instance_data_labels_alias_warns():
    with pytest.warns(UserWarning, match='labels'):
        data = BareData(labels=np.array([0, 1]))

    assert np.all(data.hypothesis == np.array([0, 1]))


def test_feature_data():
    FeatureData(features=np.ones((10, 2)), hypothesis=None)
    FeatureData(features=np.ones((10, 2)), hypothesis=np.ones((10,)))
    with pytest.raises(ValidationError):
        FeatureData(features=np.ones((10, 2)), hypothesis=np.ones((11,)))

    # illegal operation
    with pytest.raises(ValidationError):
        instances = FeatureData(features=np.ones((10, 2)), hypothesis=np.ones((10,)))
        instances.features = np.ones((10, 2))

    # initializing FeatureData with non-numeric feature values is an error
    with pytest.raises(ValidationError):
        FeatureData(features=np.array(['1'] * 10), hypothesis=np.ones((10,)))


def test_concatenate():
    data = FeatureData(features=np.ones((10, 2)))
    assert concatenate_instances(data, data) == FeatureData(features=np.ones((20, 2)))

    data = FeatureData(features=np.ones((10, 2)), extra1=3, extra2=None)
    assert concatenate_instances(data, data) == FeatureData(features=np.ones((20, 2)), extra1=3, extra2=None)

    data = FeatureData(features=np.ones((10, 2)), extra1=[1, 2])
    assert concatenate_instances(data, data) == FeatureData(features=np.ones((20, 2)), extra1=[1, 2])

    assert concatenate_instances(
        FeatureData(features=np.ones((10, 2)), extra1=3), FeatureData(features=np.ones((10, 2)), extra1=4)
    ) == FeatureData(features=np.ones((20, 2)), extra1=None)

    assert concatenate_instances(
        FeatureData(features=np.ones((10, 2)), extra1=[1, 2]), FeatureData(features=np.ones((10, 2)), extra1=[2, 1])
    ) == FeatureData(features=np.ones((20, 2)), extra1=None)

    with pytest.raises(ValueError):
        concatenate_instances(
            FeatureData(features=np.ones((10, 2)), extra1=[1, 2]), FeatureData(features=np.ones((10, 2)))
        )


def test_pair_data():
    """
    Check consistency and validation mechanism of `PairedFeatureData`.
    """
    PairedFeatureData(features=np.ones((10, 9, 1)), n_trace_instances=4, n_ref_instances=5)

    with pytest.raises(ValueError):
        PairedFeatureData(features=np.ones((10, 9)), n_trace_instances=4, n_ref_instances=5)

    with pytest.raises(ValueError):
        PairedFeatureData(features=np.ones((10, 9, 1)), n_trace_instances=4, n_ref_instances=4)

    assert PairedFeatureData(
        features=np.ones((10, 9, 1)), n_trace_instances=4, n_ref_instances=5
    ).features_trace.shape == (10, 4, 1)
    assert PairedFeatureData(
        features=np.ones((10, 9, 1)), n_trace_instances=4, n_ref_instances=5
    ).features_ref.shape == (10, 5, 1)
    assert PairedFeatureData(
        features=np.ones((10, 9, 3, 4)), n_trace_instances=4, n_ref_instances=5
    ).features_ref.shape == (10, 5, 3, 4)


def test_sourceids():
    """
    Check consistency and validation mechanism of `PairedFeatureData`.
    """
    FeatureData(features=np.ones((10, 2)), hypothesis=np.ones(10), source_ids=np.ones((10, 1)))
    PairedFeatureData(
        features=np.ones((10, 2, 2)),
        hypothesis=np.ones(10),
        source_ids=np.ones((10, 2)),
        n_ref_instances=1,
        n_trace_instances=1,
    )

    # invalid dimensions for source_ids
    with pytest.raises(ValueError):
        FeatureData(features=np.ones((10, 2)), hypothesis=np.ones(10), source_ids=np.ones((11,)))

    # invalid dimensions for source_ids
    with pytest.raises(ValueError):
        PairedFeatureData(
            features=np.ones((10, 2, 2)),
            hypothesis=np.ones(10),
            source_ids=np.ones((10, 1)),
            n_ref_instances=1,
            n_trace_instances=1,
        )


def test_llr_data():
    LLRData(features=np.ones((10, 1)))
    LLRData(features=np.ones((10, 3)))
    LLRData(features=np.ones((10, 1)), hypothesis=np.ones(10))

    # TODO: for now, 1d features are silently converted to 2d instead of raising an error
    # with pytest.raises(ValidationError):
    #     LLRData(features=np.ones((10,)))
    #     pytest.fail("features must have 2 dimensions")

    with pytest.raises(ValidationError):
        LLRData(features=np.ones((10, 2)))
        pytest.fail('LLRs must have 1 column (without interval) or 3 columns (with interval)')

    with pytest.raises(ValidationError):
        LLRData(features=np.ones((10, 4)))
        pytest.fail('LLRs must have 1 column (without interval) or 3 columns (with interval)')

    with pytest.raises(ValidationError):
        LLRData(features=np.ones((10, 3, 1)))
        pytest.fail('invalid dimensions for features')

    with pytest.raises(ValidationError):
        LLRData(features=np.ones((10,)), hypothesis=np.ones(11))
        pytest.fail('dimensions do not match')

    llr_values = np.arange(30).reshape(10, 3)
    with pytest.raises(ValidationError):
        LLRData(features=llr_values)
        pytest.fail('llrs outside their intervals')

    llr_values = llr_values[:, [1, 0, 2]]  # rearrange columns
    assert np.all(LLRData(features=llr_values).llrs == llr_values[:, 0])
    assert np.all(LLRData(features=llr_values[:, 0:1]).llrs == llr_values[:, 0])
    assert np.all(LLRData(features=llr_values).llr_intervals == llr_values[:, 1:3])


def test_indexing():
    features = FeatureData(features=np.ones((10, 1)), hypothesis=np.ones(10))
    assert len(features[:2]) == 2
    assert len(features[1]) == 1
    assert len(features[-1]) == 1


def test_concatenate_field_with_numpy_arrays():
    # Concatenate fields should work for numpy arrays
    class TestData(InstanceData):
        field: np.ndarray

        def __len__(self) -> int:
            return 1

    data1 = TestData(field=np.array([1, 2]))
    data2 = TestData(field=np.array([1, 2]))
    data3 = TestData(field=np.array([3, 4]))

    assert np.array_equal(TestData._concatenate_field('field', [data1.field, data2.field]), np.array([1, 2, 1, 2]))
    assert np.array_equal(
        TestData._concatenate_field('field', [data1.field, data2.field, data3.field]), np.array([1, 2, 1, 2, 3, 4])
    )


def test_concatenate_field():
    # Concatenate fields should work for dict, list, and numpy array types
    class TestData(InstanceData):
        field0: int
        field1: list[int]
        field2: dict[str, int]
        field3: Any

        def __len__(self) -> int:
            return 1

    data1 = TestData(field0=1, field1=[1, 2], field2={'a': 1, 'b': 2}, field3={'a1': np.array([1, 2])})
    data2 = TestData(field0=1, field1=[1, 2], field2={'b': 2, 'a': 1}, field3={'a1': np.array([1, 2])})
    data3 = TestData(field0=2, field1=[1, 3], field2={'b': 2, 'c': 3}, field3={'a1': {'b': 1}})

    assert TestData._concatenate_field('field0', [data1.field0, data2.field0]) == 1
    assert TestData._concatenate_field('field0', [data1.field0, data2.field0, data3.field0]) is None

    assert TestData._concatenate_field('field1', [data1.field1, data2.field1]) == [1, 2]
    assert TestData._concatenate_field('field1', [data1.field1, data2.field1, data3.field1]) is None

    assert TestData._concatenate_field('field2', [data1.field2, data2.field2]) == {'a': 1, 'b': 2}
    assert TestData._concatenate_field('field2', [data1.field2, data2.field2, data3.field2]) is None

    # When comapring numpy arrays, == does not work. Thus None is returned.
    # Note that numpy arrays are only compared if they are inside some other data structure (like a dict or list).
    # If the field itself is a numpy array, then the concatenation works (see test_concatenate_field_with_numpy_arrays).
    assert TestData._concatenate_field('field3', [data1.field3, data2.field3]) is None
    assert TestData._concatenate_field('field3', [data1.field3, data2.field3, data3.field3]) is None
