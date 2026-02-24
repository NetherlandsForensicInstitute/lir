import numpy as np

from lir.data.models import PairedFeatureData
from lir.transform.distance import ElementWiseDifference, EuclideanDistance, ManhattanDistance


def test_element_wise():
    dist = ElementWiseDifference()
    features = PairedFeatureData(
        features=np.stack([np.ones((10, 100)), np.ones((10, 100)) * 2], axis=1), n_trace_instances=1, n_ref_instances=1
    )
    assert features.features.shape == (10, 2, 100)
    assert dist.apply(features).features.shape == (10, 100)
    assert np.all(dist.apply(features).features == np.ones((10, 100)))


def test_manhattan():
    manhattan = ManhattanDistance()
    features = PairedFeatureData(
        features=np.stack([np.ones((10, 100)), np.ones((10, 100)) * 2], axis=1), n_trace_instances=1, n_ref_instances=1
    )

    assert features.features.shape == (10, 2, 100)
    assert manhattan.apply(features).features.shape == (10, 1)
    assert np.all(manhattan.apply(features).features == np.ones((10, 1)) * 100)

    element_wise_diff = ElementWiseDifference().apply(features)
    assert np.all(manhattan.apply(element_wise_diff).features == np.ones((10, 1)) * 100)


def test_euclidean():
    euclidean_distance = EuclideanDistance()
    features = PairedFeatureData(
        features=np.stack([np.ones((10, 100)), np.ones((10, 100)) * 2], axis=1), n_trace_instances=1, n_ref_instances=1
    )

    assert features.features.shape == (10, 2, 100)
    assert euclidean_distance.apply(features).features.shape == (10, 1)
    assert np.all(euclidean_distance.apply(features).features == np.ones((10, 1)) * np.sqrt(100))

    element_wise_diff = ElementWiseDifference().apply(features)
    assert np.all(euclidean_distance.apply(element_wise_diff).features == np.ones((10, 1)) * np.sqrt(100))
