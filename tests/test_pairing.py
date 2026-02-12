import unittest

import numpy as np
import pytest

from lir.data.models import FeatureData
from lir.datasets.synthesized_normal_multiclass import SynthesizedDimension, SynthesizedNormalMulticlassData
from lir.transform.pairing import InstancePairing, SourcePairing


def test_instance_pairing_seed():
    dimensions = [SynthesizedDimension(population_mean=0, population_std=5, sources_std=1)]
    data = SynthesizedNormalMulticlassData(dimensions, population_size=100, sources_size=2, seed=0)

    pairing0 = InstancePairing(seed=1)
    pairing1 = InstancePairing(seed=1)
    instances = data.get_instances()
    assert pairing0.pair(instances) == pairing1.pair(instances)

    pairing0 = InstancePairing(seed=1, ratio_limit=1)
    pairing1 = InstancePairing(seed=1, ratio_limit=1)
    assert pairing0.pair(instances) == pairing0.pair(instances)
    assert pairing0.pair(instances) == pairing1.pair(instances)


@pytest.mark.parametrize(
    'pairing,n_pairs_found,features,source_ids,n_trace_instances,n_ref_instances',
    [
        (
            np.array([[0, 1]]),  # pair indices
            1,
            np.ones((3, 1)),  # features
            np.arange(3),  # source_ids
            1,
            1,
        ),
        (
            np.array([[0, 1], [0, 2]]),  # pair indices
            2,
            np.ones((3, 1)),  # features
            np.arange(3),  # source_ids
            1,
            1,
        ),
        (
            np.array([[0, 0]]),  # pair indices
            0,
            np.ones((3, 1)),
            np.arange(3),  # source_ids
            1,
            1,
        ),
        (
            np.array([[0, 0]]),  # pair indices
            1,
            np.ones((10, 1)),  # features
            np.zeros(10),  # source_ids
            1,
            1,
        ),
        (
            np.array([[0, 0], [1, 1]]),  # pair indices
            1,
            np.ones((10, 1)),  # features
            np.zeros(10),  # source_ids
            1,
            1,
        ),
        (
            np.array([[0, 0]]),  # pair indices
            1,
            np.ones((10, 1)),  # features
            np.zeros(10),  # source_ids
            4,
            6,
        ),
        (
            np.array([[0, 0]]),  # pair indices
            0,
            np.ones((10, 1)),  # features
            np.zeros(10),  # source_ids
            5,
            6,
        ),
    ],
)
def test_construct_array(
    pairing: np.ndarray,
    n_pairs_found: int,
    features: np.ndarray,
    source_ids: np.ndarray,
    n_trace_instances: int,
    n_ref_instances: int,
):
    instances = FeatureData(features=features, source_ids=source_ids, meta=features)
    paired_data = SourcePairing()._construct_array(
        pairing,
        instances,
        n_trace_instances,
        n_ref_instances,
    )
    assert n_pairs_found == len(paired_data), 'number of output pairs'
    assert (
        len(paired_data) == 0
        or paired_data.features.shape[1] == paired_data.meta.shape[1] == n_trace_instances + n_ref_instances
    )
    assert paired_data.features.shape[2] == instances.features.shape[1]
    assert len(paired_data) == 0 or paired_data.meta.shape[2] == instances.meta.shape[1]

    target_shape = (n_pairs_found, n_trace_instances + n_ref_instances) + instances.features.shape[1:]
    assert paired_data.features.shape == target_shape


@pytest.mark.parametrize(
    'n_pairs_expected,features,source_ids,n_trace_instances,n_ref_instances',
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
    source_ids: np.ndarray,
    n_trace_instances: int,
    n_ref_instances: int,
):
    instances = FeatureData(features=features, source_ids=source_ids)
    paired_data = SourcePairing().pair(instances, n_trace_instances, n_ref_instances)
    assert len(paired_data) == n_pairs_expected
    assert np.all(
        paired_data.features.shape
        == np.array((n_pairs_expected, n_trace_instances + n_ref_instances) + instances.features.shape[1:])
    )


class TestPairing(unittest.TestCase):
    instances = FeatureData(
        features=np.arange(30).reshape(10, 3), source_ids=np.concatenate([np.arange(5), np.arange(5)])
    )

    def test_pairing1(self):
        pairing = InstancePairing()
        pairs = pairing.pair(self.instances)

        self.assertEqual(np.sum(pairs.labels == 1), 5, 'number of same source pairs')
        self.assertEqual(
            np.sum(pairs.labels == 0),
            2 * (8 + 6 + 4 + 2),
            'number of different source pairs',
        )

    def test_pairing2(self):
        pairing = InstancePairing(ratio_limit=1)
        pairs = pairing.pair(self.instances)

        self.assertEqual(np.sum(pairs.labels == 1), 5, 'number of same source pairs')
        self.assertEqual(np.sum(pairs.labels == 0), 5, 'number of different source pairs')

        assert np.all(pairs.instance_indices[:, 0] != pairs.instance_indices[:, 1]), 'identity in pairs'

    def test_pairing_ratio1(self):
        # test ratio
        ratio_limit = 7
        pairing_ratio = InstancePairing(ratio_limit=ratio_limit)
        pairs = pairing_ratio.pair(self.instances)
        ratio = np.sum(pairs.labels == 0) / np.sum(pairs.labels == 1)
        self.assertEqual(ratio, ratio_limit, 'ratio ss ds pairs not correct')

    def test_pairing_ratio2(self):
        # if ratio_limit exceeds highest possible ratio, all ds pairs are selected
        ratio_limit = 1_000
        pairing_ratio_max = InstancePairing(ratio_limit=ratio_limit)
        pairs = pairing_ratio_max.pair(self.instances)
        ratio = np.sum(pairs.labels == 0) / np.sum(pairs.labels == 1)
        self.assertLess(
            ratio,
            ratio_limit,
            'realised ratio should be less than or equal to ratio_limit',
        )
        self.assertEqual(
            np.sum(pairs.labels == 0),
            2 * (8 + 6 + 4 + 2),
            'all different source pairs should be selected',
        )

    def test_pairing_ratio3(self):
        # test ratio with same_source_limit
        ratio_limit = 5
        same_source_limit = 3
        pairing_ratio_ss_lim = InstancePairing(ratio_limit=ratio_limit, same_source_limit=same_source_limit)
        pairs = pairing_ratio_ss_lim.pair(self.instances)
        ratio = np.sum(pairs.labels == 0) / np.sum(pairs.labels == 1)
        self.assertEqual(ratio, ratio_limit, 'ratio ss ds pairs not correct')
        self.assertEqual(same_source_limit, np.sum(pairs.labels == 1), 'ss pairs limit')

    def test_pairing_ratio4(self):
        # test ratio with different_source_limit
        ratio_limit = 5
        different_source_limit = 20
        pairing_ratio_ds_lim = InstancePairing(ratio_limit=ratio_limit, different_source_limit=different_source_limit)
        pairs = pairing_ratio_ds_lim.pair(self.instances)
        ratio = np.sum(pairs.labels == 0) / np.sum(pairs.labels == 1)
        self.assertLessEqual(ratio, ratio_limit, 'ratio ss ds pairs not correct')
        self.assertLessEqual(np.sum(pairs.labels == 0), different_source_limit, 'ds pairs limit')

    def test_pairing_ratio5(self):
        # test ratio with same_source_limit and different_source_limit
        ratio_limit = 5
        same_source_limit = 9
        different_source_limit = 30
        pairing_ratio_ds_lim = InstancePairing(
            ratio_limit=ratio_limit,
            same_source_limit=same_source_limit,
            different_source_limit=different_source_limit,
        )
        pairs = pairing_ratio_ds_lim.pair(self.instances)
        ratio = np.sum(pairs.labels == 0) / np.sum(pairs.labels == 1)
        self.assertLessEqual(ratio, ratio_limit, 'ratio_limit ss and ds pairs exceeded')
        self.assertLessEqual(np.sum(pairs.labels == 1), same_source_limit, 'ss pairs limit')
        self.assertLessEqual(np.sum(pairs.labels == 0), different_source_limit, 'ds pairs limit')

    def test_pairing_seed(self):
        pairing_seed_1 = InstancePairing(ratio_limit=1, seed=123)
        pairs1 = pairing_seed_1.pair(self.instances)

        pairing_seed_2 = InstancePairing(ratio_limit=1, seed=123)
        pairs2 = pairing_seed_2.pair(self.instances)

        pairing_seed_3 = InstancePairing(ratio_limit=1, seed=456)
        pairs3 = pairing_seed_3.pair(self.instances)

        assert np.all(pairs1.features == pairs2.features), 'same seed, same X pairs'
        assert np.any(pairs1.features != pairs3.features), 'different seed, different X pairs'
