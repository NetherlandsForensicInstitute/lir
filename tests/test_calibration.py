import math
import unittest
import warnings

import numpy as np

from lir.algorithms.isotonic_regression import IsotonicCalibrator
from lir.algorithms.logistic_regression import LogitCalibrator
from lir.util import (
    Xn_to_Xy,
    Xy_to_Xn,
    logodds_to_odds,
    odds_to_probability,
    probability_to_logodds,
)


warnings.simplefilter('error')


def _cllr(lr0, lr1):
    with np.errstate(divide='ignore'):
        cllr0 = np.mean(np.log2(1 + lr0))
        cllr1 = np.mean(np.log2(1 + 1 / lr1))
        return 0.5 * (cllr0 + cllr1)


def _pdf(X, mu, sigma):
    return np.exp(-np.power(X - mu, 2) / (2 * sigma * sigma)) / math.sqrt(2 * math.pi * sigma * sigma)


class TestIsotonicRegression(unittest.TestCase):
    def test_lr_1(self):
        score_class0 = np.arange(0, 1, 0.1)
        score_class1 = np.arange(0, 1, 0.1)
        X, y = Xn_to_Xy(score_class0, score_class1)
        irc = IsotonicCalibrator()
        llrs = irc.fit_transform(probability_to_logodds(X), y)
        lr0, lr1 = Xy_to_Xn(logodds_to_odds(llrs), y)
        self.assertEqual(score_class0.shape, lr0.shape)
        self.assertEqual(score_class1.shape, lr1.shape)
        np.testing.assert_almost_equal(lr0, [1.0] * lr0.shape[0])
        np.testing.assert_almost_equal(lr1, [1.0] * lr1.shape[0])

    def run_cllrmin(self, lr0, lr1, places=7):
        lr0 = np.array(lr0)
        lr1 = np.array(lr1)
        X, y = Xn_to_Xy(lr0, lr1)
        cllr = _cllr(lr0, lr1)

        irc = IsotonicCalibrator()
        llrs = irc.fit_transform(odds_to_probability(X), y)
        lrmin0, lrmin1 = Xy_to_Xn(logodds_to_odds(llrs), y)

        cllrmin = _cllr(lrmin0, lrmin1)

        self.assertAlmostEqual(cllr, cllrmin, places=places)

    def test_cllrmin(self):
        self.run_cllrmin([1] * 10, [1] * 10)
        self.run_cllrmin([1], [1] * 10)
        self.run_cllrmin([4, 0.25, 0.25, 0.25, 0.25, 1], [4, 4, 4, 4, 0.25, 1])

        # np.random.seed(0)
        X0 = np.random.normal(loc=0, scale=1, size=(40000,))
        X1 = np.random.normal(loc=1, scale=1, size=(40000,))
        lr0 = _pdf(X0, 1, 1) / _pdf(X0, 0, 1)
        lr1 = _pdf(X1, 1, 1) / _pdf(X1, 0, 1)
        self.run_cllrmin(lr0, lr1, places=2)
        self.run_cllrmin(lr0, lr1[:30000], places=2)

    def test_lr_almost_1(self):
        score_class0 = np.arange(0, 1, 0.1)
        score_class1 = np.arange(0.05, 1.05, 0.1)
        X, y = Xn_to_Xy(score_class0, score_class1)
        irc = IsotonicCalibrator()
        llrs = irc.fit_transform(X, y)
        lr0, lr1 = Xy_to_Xn(logodds_to_odds(llrs), y)
        self.assertEqual(score_class0.shape, lr0.shape)
        self.assertEqual(score_class1.shape, lr1.shape)
        np.testing.assert_almost_equal(lr0, np.concatenate([[0], [1.0] * (lr0.shape[0] - 1)]))
        np.testing.assert_almost_equal(lr1, np.concatenate([[1.0] * (lr1.shape[0] - 1), [np.inf]]))


class TestLogitCalibrator(unittest.TestCase):
    score_class0 = [
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
    score_class1 = [
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

    def test_prob_version(self):
        X, y = Xn_to_Xy(self.score_class0, self.score_class1)
        X = odds_to_probability(X)
        X = probability_to_logodds(X)
        desired = [
            0.1794121972972742,
            0.00041466046383887233,
            0.0002893188549697442,
            0.00025706927831129677,
            0.08740293107490102,
            0.136624384190492,
            0.0119387233899553,
            2.5306657472748792e-05,
            0.000585723303153669,
            584.0144216917622,
            72.58450757957488,
            36.25831270280195,
            30.37802498043253,
            3.593416485014456,
            24.124059461659918,
            45.02979931421313,
            73.83453694601398,
            4744.539884048114,
        ]

        calibrator = LogitCalibrator()
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
            0.17938619741047357,
            0.0004145423988770249,
            0.00028923408006262664,
            0.0002569932531617985,
            0.08738881700201688,
            0.13660372753636074,
            0.011936248051671832,
            2.529782221300445e-05,
            0.0005855611924232705,
            0.0,
            1.7898780873341496e-177,
            0.0,
            584.0381138155668,
            72.58397448484948,
            36.25750145609919,
            30.377205489827194,
            3.593143814999971,
            24.12328565793142,
            45.02902685404331,
            73.83412361274337,
            4751.481164586008,
            np.inf,
            350076322.00372607,
            np.inf,
        ]

        calibrator = LogitCalibrator()
        calibrator.fit(X, y)
        llrs_cal = calibrator.transform(X)
        np.testing.assert_allclose(logodds_to_odds(llrs_cal), desired, rtol=1e-2)


if __name__ == '__main__':
    unittest.main()
