import unittest

import numpy as np
import pytest

from lir.transform.pairing import InstancePairing, SourcePairing
from lir.data.datasets.synthesized_normal_multiclass import (
    SynthesizedNormalMulticlassData,
    SynthesizedDimension,
)


def test_instance_pairing_seed():
    dimensions = [
        SynthesizedDimension(population_mean=0, population_std=5, sources_std=1)
    ]
    data = SynthesizedNormalMulticlassData(
        dimensions, population_size=100, sources_size=2, seed=0
    )

    pairing0 = InstancePairing()
    pairing1 = InstancePairing()
    for values in zip(
        pairing0.pair(*data.get_instances()), pairing1.pair(*data.get_instances())
    ):
        np.all(values[0] == values[1])

    pairing0 = InstancePairing(ratio_limit=1)
    pairing1 = InstancePairing(ratio_limit=1)
    for values in zip(
        pairing0.pair(*data.get_instances()), pairing0.pair(*data.get_instances())
    ):
        np.all(values[0] == values[1])
    for values in zip(
        pairing0.pair(*data.get_instances()), pairing1.pair(*data.get_instances())
    ):
        np.all(values[0] == values[1])


@pytest.mark.parametrize(
    "pairing,n_pairs_found,features,labels,n_trace_instances,n_ref_instances",
    [
        (
            np.array([[0, 1]]),  # pair indices
            1,
            np.ones((3, 1)),  # features
            np.arange(3),  # labels
            1,
            1,
        ),
        (
            np.array([[0, 1], [0, 2]]),  # pair indices
            2,
            np.ones((3, 1)),  # features
            np.arange(3),  # labels
            1,
            1,
        ),
        (
            np.array([[0, 0]]),  # pair indices
            0,
            np.ones((3, 1)),
            np.arange(3),
            1,
            1,
        ),
        (
            np.array([[0, 0]]),  # pair indices
            1,
            np.ones((10, 1)),  # features
            np.zeros(10),  # labels
            1,
            1,
        ),
        (
            np.array([[0, 0], [1, 1]]),  # pair indices
            1,
            np.ones((10, 1)),  # features
            np.zeros(10),  # labels
            1,
            1,
        ),
        (
            np.array([[0, 0]]),  # pair indices
            1,
            np.ones((10, 1)),  # features
            np.zeros(10),  # labels
            4,
            6,
        ),
        (
            np.array([[0, 0]]),  # pair indices
            0,
            np.ones((10, 1)),  # features
            np.zeros(10),  # labels
            5,
            6,
        ),
    ],
)
def test_construct_array(
    pairing: np.ndarray,
    n_pairs_found: int,
    features: np.ndarray,
    labels: np.ndarray,
    n_trace_instances: int,
    n_ref_instances: int,
):
    meta = features
    pair_features, pair_labels, pair_meta = SourcePairing()._construct_array(
        pairing,
        features,
        labels,
        meta,
        n_trace_instances,
        n_ref_instances,
    )
    assert (
        n_pairs_found
        == pair_features.shape[0]
        == pair_labels.shape[0]
        == pair_meta.shape[0]
    ), "number of output pairs"
    assert (
        pair_features.shape[1]
        == pair_meta.shape[1]
        == n_trace_instances + n_ref_instances
    )
    assert pair_features.shape[2] == features.shape[1]
    assert pair_meta.shape[2] == meta.shape[1]
    assert np.all(
        pair_features.shape
        == np.array(
            (n_pairs_found, n_trace_instances + n_ref_instances) + features.shape[1:]
        )
    )


@pytest.mark.parametrize(
    "n_pairs_expected,features,labels,n_trace_instances,n_ref_instances",
    [
        (
            3,
            np.ones((3, 1)),  # features
            np.arange(3),  # labels
            1,
            1,
        ),
        (
            3,
            np.ones((3, 9)),  # features
            np.arange(3),  # labels
            1,
            1,
        ),
        (
            6,
            np.ones((6, 9)),  # features
            np.repeat(np.arange(3), 2),  # labels
            1,
            1,
        ),
        (
            6,
            np.ones((18, 7)),  # features
            np.repeat(np.arange(3), 6),  # labels
            3,
            3,
        ),
        (
            3,
            np.ones((18, 7)),  # features
            np.repeat(np.arange(3), 6),  # labels
            3,
            4,
        ),
        (
            0,
            np.ones((18, 7)),  # features
            np.repeat(np.arange(3), 6),  # labels
            7,
            1,
        ),
    ],
)
def test_source_level_pairing(
    n_pairs_expected: int,
    features: np.ndarray,
    labels: np.ndarray,
    n_trace_instances: int,
    n_ref_instances: int,
):
    pair_features, pair_labels, pair_meta = SourcePairing().pair(
        features, labels, features, n_trace_instances, n_ref_instances
    )
    assert len(pair_labels) == n_pairs_expected
    assert np.all(
        pair_features.shape
        == np.array(
            (n_pairs_expected, n_trace_instances + n_ref_instances) + features.shape[1:]
        )
    )


class TestPairing(unittest.TestCase):
    X = np.arange(30).reshape(10, 3)
    y = np.concatenate([np.arange(5), np.arange(5)])

    def test_pairing(self):
        pairing = InstancePairing()
        X_pairs, y_pairs = pairing._transform(self.X, self.y)

        self.assertEqual(np.sum(y_pairs == 1), 5, "number of same source pairs")
        self.assertEqual(
            np.sum(y_pairs == 0),
            2 * (8 + 6 + 4 + 2),
            "number of different source pairs",
        )

        pairing = InstancePairing(ratio_limit=1)
        X_pairs, y_pairs = pairing._transform(self.X, self.y)

        self.assertEqual(np.sum(y_pairs == 1), 5, "number of same source pairs")
        self.assertEqual(np.sum(y_pairs == 0), 5, "number of different source pairs")

        self.assertTrue(
            np.all(pairing.pairing[:, 0] != pairing.pairing[:, 1]), "identity in pairs"
        )

    def test_pairing_ratio(self):
        # test ratio
        ratio_limit = 7
        pairing_ratio = InstancePairing(ratio_limit=ratio_limit)
        X_pairs, y_pairs = pairing_ratio._transform(self.X, self.y)
        ratio = np.sum(y_pairs == 0) / np.sum(y_pairs == 1)
        self.assertEqual(ratio, ratio_limit, "ratio ss ds pairs not correct")

        # if ratio_limit exceeds highest possible ratio, all ds pairs are selected
        ratio_limit = 1_000
        pairing_ratio_max = InstancePairing(ratio_limit=ratio_limit)
        X_pairs, y_pairs = pairing_ratio_max._transform(self.X, self.y)
        ratio = np.sum(y_pairs == 0) / np.sum(y_pairs == 1)
        self.assertLess(
            ratio,
            ratio_limit,
            "realised ratio should be less or equal than ratio_limit",
        )
        self.assertEqual(
            np.sum(y_pairs == 0),
            2 * (8 + 6 + 4 + 2),
            "all different source pairs should be selected",
        )

        # test ratio with same_source_limit
        ratio_limit = 5
        same_source_limit = 3
        pairing_ratio_ss_lim = InstancePairing(
            ratio_limit=ratio_limit, same_source_limit=same_source_limit
        )
        X_pairs, y_pairs = pairing_ratio_ss_lim._transform(self.X, self.y)
        ratio = np.sum(y_pairs == 0) / np.sum(y_pairs == 1)
        self.assertEqual(ratio, ratio_limit, "ratio ss ds pairs not correct")
        self.assertEqual(same_source_limit, np.sum(y_pairs == 1), "ss pairs limit")

        # test ratio with different_source_limit
        ratio_limit = 5
        different_source_limit = 20
        pairing_ratio_ds_lim = InstancePairing(
            ratio_limit=ratio_limit, different_source_limit=different_source_limit
        )
        X_pairs, y_pairs = pairing_ratio_ds_lim._transform(self.X, self.y)
        ratio = np.sum(y_pairs == 0) / np.sum(y_pairs == 1)
        self.assertLessEqual(ratio, ratio_limit, "ratio ss ds pairs not correct")
        self.assertLessEqual(
            np.sum(y_pairs == 0), different_source_limit, "ds pairs limit"
        )

        # test ratio with same_source_limit and different_source_limit
        same_source_limit = 9
        different_source_limit = 30
        pairing_ratio_ds_lim = InstancePairing(
            ratio_limit=ratio_limit,
            same_source_limit=same_source_limit,
            different_source_limit=different_source_limit,
        )
        X_pairs, y_pairs = pairing_ratio_ds_lim._transform(self.X, self.y)
        ratio = np.sum(y_pairs == 0) / np.sum(y_pairs == 1)
        self.assertLessEqual(ratio, ratio_limit, "ratio_limit ss and ds pairs exceeded")
        self.assertLessEqual(np.sum(y_pairs == 1), same_source_limit, "ss pairs limit")
        self.assertLessEqual(
            np.sum(y_pairs == 0), different_source_limit, "ds pairs limit"
        )

    def test_pairing_seed(self):
        pairing_seed_1 = InstancePairing(ratio_limit=1, seed=123)
        X_pairs_1, y_pairs_1 = pairing_seed_1._transform(self.X, self.y)

        pairing_seed_2 = InstancePairing(ratio_limit=1, seed=123)
        X_pairs_2, y_pairs_2 = pairing_seed_2._transform(self.X, self.y)

        pairing_seed_3 = InstancePairing(ratio_limit=1, seed=456)
        X_pairs_3, y_pairs_3 = pairing_seed_3._transform(self.X, self.y)

        self.assertTrue(np.all(X_pairs_1 == X_pairs_2), "same seed, same X pairs")
        self.assertTrue(
            np.any(X_pairs_1 != X_pairs_3), "different seed, different X pairs"
        )
