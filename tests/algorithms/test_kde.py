import math
import unittest

import numpy as np
import pytest

from lir.algorithms.kde import KDECalibrator, parse_bandwidth
from lir.util import Xn_to_Xy, logodds_to_odds, odds_to_probability, probability_to_logodds


def test_kde_dimensions():
    features = np.random.normal(loc=0, scale=1, size=(10, 1))
    labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    kde = KDECalibrator(bandwidth=1)
    kde.fit(features, labels).transform(features)
    kde.fit(features.flatten(), labels).transform(features.flatten())


# the X-data for TestKDECalibrator,  TestGaussianCalibrator, TestLogitCalibrator comes from random draws of perfectly
# calibrated LLR-distributions with mu_s = 6. For larger datasets it is confirmed that the calibration function
# approaches the line Y = X. The data under H1 and H0 are the 1:10 elements.
class TestKDECalibrator(unittest.TestCase):
    score_class0 = np.array(
        [
            9.10734621e-02,
            1.37045394e-06,
            7.09420701e-07,
            5.71489514e-07,
            2.44360004e-02,
            5.53264987e-02,
            6.40338659e-04,
            8.22553310e-09,
            2.57792725e-06,
        ]
    )
    score_class1 = np.array(
        [
            2.42776744e05,
            5.35255527e03,
            1.50355963e03,
            1.08776892e03,
            2.19083530e01,
            7.13508826e02,
            2.23486401e03,
            5.52239060e03,
            1.12077833e07,
        ]
    )

    def test_kde_calibrator(self):
        X, y = Xn_to_Xy(self.score_class0, self.score_class1)
        X = odds_to_probability(X)
        X = probability_to_logodds(X)
        desired = [
            3.59562799e-02,
            1.75942116e-11,
            2.59633540e-12,
            1.36799721e-12,
            8.15673411e-03,
            2.10030624e-02,
            3.70456430e-05,
            1.40710861e-18,
            1.04459592e-10,
            3.14589737e03,
            2.59568527e02,
            1.08519904e02,
            8.56459139e01,
            3.81243702e00,
            6.23873841e01,
            1.43844114e02,
            2.64913149e02,
            1.49097168e05,
        ]
        calibrator = KDECalibrator(bandwidth='silverman')
        calibrator.fit(X, y)
        llrs_cal = calibrator.transform(X)
        np.testing.assert_allclose(logodds_to_odds(llrs_cal), desired)

    def test_on_extreme_values(self):
        X = np.array(
            [
                8.34714300e-002,
                1.37045206e-006,
                7.09420198e-007,
                5.71489187e-007,
                2.38531254e-002,
                5.24259542e-002,
                6.39928887e-004,
                8.22553304e-009,
                2.57792061e-006,
                0.00000000e000,
                9.88131292e-324,
                0.00000000e000,
                9.99995881e-001,
                9.99813208e-001,
                9.99335354e-001,
                9.99081531e-001,
                9.56347800e-001,
                9.98600437e-001,
                9.99552746e-001,
                9.99818952e-001,
                9.99999911e-001,
                1.00000000e000,
                1 - np.float_power(10, -16),
                1.00000000e000,
            ]
        )
        X = probability_to_logodds(X)
        y = np.concatenate((np.zeros(12), np.ones(12)))
        desired = [
            6.148510640582358,
            0.10548096579142373,
            0.07571171879632102,
            0.06774859414831141,
            4.408883097248305,
            5.446103603204983,
            1.4258427450086562,
            0.006102474459494191,
            0.14360453961912525,
            0.0,
            0.0,
            0.0,
            17.786943105214274,
            21.248067409078676,
            21.10676921763807,
            20.955468109356307,
            16.029054988277238,
            20.689727349181517,
            21.22851434841379,
            21.24246276550688,
            11.31919250180751,
            math.inf,
            2.846712755553574,
            math.inf,
        ]
        calibrator = KDECalibrator(bandwidth='silverman')
        calibrator.fit(X, y)
        llrs_cal = calibrator.transform(X)
        np.testing.assert_allclose(logodds_to_odds(llrs_cal), desired)


@pytest.mark.parametrize(
    'bandwidth_definition,expected_bandwidth',
    [
        ([1, 2], (1, 2)),  # list input (`Sized` type)
        ((1, 2), (1, 2)),  # tuple input (`Sized` type)
        (lambda X, y: (1234, 5678), (1234, 5678)),  # callable function
        ('silverman', (1.05922384, 0.46105395)),  # Silverman algorithm
        (123.45, (123.45, 123.45)),  # float
        (12, (12, 12)),  # integer
    ],
)
def test_kde_bandwidth_parsing_supported_types(bandwidth_definition, expected_bandwidth):
    bandwidth_fn = parse_bandwidth(bandwidth_definition)

    samples_one_feature = np.asarray([[1], [2], [3]])
    labels = np.asarray([0, 1, 1])

    bandwidth = bandwidth_fn(X=samples_one_feature, y=labels)

    assert np.allclose(bandwidth, expected_bandwidth)


@pytest.mark.parametrize(
    'bandwidth_definition',
    [
        None,  # undefined
        (1, 2, 3),  # tuples of size 3
        [1, 2, 3],  # lists of size 3
        (1,),  # tuples of size 1
        [1],  # lists of size 1
        'inexistent',  # unsupported bandwidth methods
        {1, 2},  # set
        {'a': 1, 'b': 2},  # dict
    ],
)
def test_kde_bandwidth_parsing_unsupported_types(bandwidth_definition):
    with pytest.raises(ValueError):
        parse_bandwidth(bandwidth_definition)
