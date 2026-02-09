import unittest

import numpy as np
import pytest

from lir import metrics
from lir.data.models import LLRData
from lir.metrics.devpav import _calcsurface, _devpavcalculator, devpav
from lir.util import Xn_to_Xy, odds_to_logodds


@pytest.mark.parametrize(
    'expected,h1_lrs,h2_lrs',
    [
        (1, [1, 1], [1, 1]),
        (2, [3.0] * 2, [1 / 3.0] * 2),
        (2, [3.0] * 20, [1 / 3.0] * 20),
        (0.4150374992788437, [1 / 3.0] * 2, [3.0] * 2),
        (0.7075187496394219, [1 / 3.0] * 2, [1]),
        (0.507177646488535, [1 / 100.0] * 100, [1]),
        (0.5400680236656377, [1 / 100.0] * 100 + [100], [1]),
        (0.5723134914863265, [1 / 100.0] * 100 + [100] * 2, [1]),
        (0.6952113122368764, [1 / 100.0] * 100 + [100] * 6, [1]),
        (1.0000000000000000, [1], [1]),
        (1.0849625007211563, [2], [2] * 2),
        (1.6699250014423126, [8], [8] * 8),
    ],
)
def test_calculate_cllr(expected: float, h1_lrs: list[float], h2_lrs: list[float]):
    lrs, labels = Xn_to_Xy(np.array(h1_lrs), np.array(h2_lrs))
    llrs = odds_to_logodds(lrs)
    llr_data = LLRData(features=llrs, labels=labels)
    pytest.approx(expected, metrics.cllr(llr_data))


@pytest.mark.parametrize(
    'expected,h1_lrs,h2_lrs',
    [
        (1, [1, 1], [1, 1]),
    ],
)
def test_calculate_cllr_min(expected: float, h1_lrs: list[float], h2_lrs: list[float]):
    lrs, labels = Xn_to_Xy(np.array(h1_lrs), np.array(h2_lrs))
    llrs = odds_to_logodds(lrs)
    llr_data = LLRData(features=llrs, labels=labels)
    pytest.approx(expected, metrics.cllr_min(llr_data))


@pytest.mark.parametrize(
    'expected,h1_llrs,h2_llrs',
    [
        (np.inf, [np.inf, 0], [0, 0]),
        (np.inf, [np.inf, -np.inf], [0, 0]),
        (np.inf, [0, 0], [-np.inf, 0]),
        (0.5, [-np.inf, -np.inf], [0, 0]),
        (0.5, [0, 0], [np.inf, np.inf]),
        (0, [-np.inf, -np.inf], [np.inf, np.inf]),
        (np.inf, [np.inf, np.inf], [-np.inf, -np.inf]),
        (np.inf, [0], [odds_to_logodds(1.0e-317)]),  # value near zero for which 1/value causes an overflow
    ],
)
def test_extreme_cllr(expected: float, h1_llrs: list[float], h2_llrs: list[float]):
    llrs, labels = Xn_to_Xy(np.array(h1_llrs), np.array(h2_llrs))
    llr_data = LLRData(features=llrs, labels=labels)
    pytest.approx(expected, metrics.cllr(llr_data))


@pytest.mark.parametrize(
    'h1_llrs,h2_llrs',
    [
        ([np.nan, 0], [0, 0]),
        ([0, 0], [0, np.nan]),
        ([np.nan, np.nan], [np.nan, np.nan]),
        ([0, 1, 2], []),
        ([], [0, 1, 2]),
    ],
)
def test_illegal_cllr(h1_llrs, h2_llrs):
    llrs, labels = Xn_to_Xy(np.array(h1_llrs), np.array(h2_llrs))
    llr_data = LLRData(features=llrs, labels=labels)
    assert np.isnan(metrics.cllr(llr_data))

    if np.all(np.isfinite(llrs)):  # this condition should be removed ?! --> see issue #301
        assert np.isnan(metrics.cllr_min(llr_data))


class TestDevPAV(unittest.TestCase):
    def test_devpav_error(self):
        lrs = np.ones((10, 1))
        llrs = odds_to_logodds(lrs)
        y = np.concatenate([np.ones(10)])
        with self.assertRaises(ValueError):
            devpav(LLRData(features=llrs, labels=y))

    def test_devpav(self):
        # naive system
        llrs = np.zeros((10, 1))
        y = np.concatenate([np.ones(5), np.zeros(5)])
        self.assertEqual(devpav(LLRData(features=llrs, labels=y)), 0)

        # badly calibrated naive system
        lrs = 2 * np.ones(10)
        llrs = odds_to_logodds(lrs)
        y = np.concatenate([np.ones(5), np.zeros(5)])
        self.assertEqual(devpav(LLRData(features=llrs, labels=y)), np.log10(2))

        # infinitely bad calibration
        lrs = np.array([5, 5, 5, 0.2, 0.2, 0.2, np.inf])
        llrs = odds_to_logodds(lrs)
        y = np.concatenate([np.ones(3), np.zeros(4)])
        self.assertEqual(devpav(LLRData(features=llrs, labels=y)), np.inf)

        # binary system
        lrs = np.array([5, 5, 5, 0.2, 5, 0.2, 0.2, 0.2])
        llrs = odds_to_logodds(lrs)
        y = np.concatenate([np.ones(4), np.zeros(4)])
        self.assertAlmostEqual(devpav(LLRData(features=llrs, labels=y)), (np.log10(5) - np.log10(3)) / 2)

        # somewhat normal
        lrs = np.array([6, 5, 5, 0.2, 5, 0.2, 0.2, 0.1])
        llrs = odds_to_logodds(lrs)
        y = np.concatenate([np.ones(4), np.zeros(4)])
        self.assertAlmostEqual(devpav(LLRData(features=llrs, labels=y)), (np.log10(5) - np.log10(2)) / 2)

        # test on dummy data 3 #######################
        lrs_same = (0.1, 100)
        lrs_dif = (10**-2, 10)
        lrs, y = Xn_to_Xy(lrs_dif, lrs_same)
        llrs = odds_to_logodds(lrs)
        self.assertEqual(devpav(LLRData(features=llrs, labels=y)), 0.5)


class TestDevpavcalculator(unittest.TestCase):
    def test_devpavcalculator(self):
        ## four tests on pathological PAV-transforms
        # 1 of 4: test on data where PAV-transform has a horizontal line starting at log(X) = -Inf
        lrs_same = (0, 1, 10**3)
        lrs_dif = (0.001, 2, 10**2)
        PAVresult = np.array([0.66666667, 0.66666667, 0.66666667, 0.66666667, 0.66666667, np.inf])
        lrs, y = Xn_to_Xy(lrs_dif, lrs_same)
        self.assertEqual(_devpavcalculator(lrs, PAVresult, y), np.inf)

        # 2 of 4: test on data where PAV-transform has a horizontal line ending at log(X) = Inf
        lrs_same = (0.01, 1, 10**2)
        lrs_dif = (0.001, 2, float('inf'))
        PAVresult = np.array([0.0, 1.5, 1.5, 1.5, 1.5, 1.5])
        lrs, y = Xn_to_Xy(lrs_dif, lrs_same)
        self.assertEqual(_devpavcalculator(lrs, PAVresult, y), np.inf)

        # 3 of 4: test on data where PAV-transform has a horizontal line starting at log(X) = -Inf, and another one
        # ending at log(X) = Inf
        lrs_same = (0, 1, 10**3, 10**3, 10**3, 10**3)
        lrs_dif = (0.001, 2, float('inf'))
        PAVresult = np.array([0.5, 0.5, 2, 0.5, 0.5, 2, 2, 2, 2])
        lrs, y = Xn_to_Xy(lrs_dif, lrs_same)
        self.assertEqual(_devpavcalculator(lrs, PAVresult, y), np.inf)

        # 4 of 4: test on data where lrs_same and lrs_dif are completely separated (and PAV result is a vertical line)
        lrs_same = (10**4, 10**5, float('inf'))
        lrs_dif = (0, 1, 10**3)
        PAVresult = np.array([0, 0, 0, float('inf'), float('inf'), float('inf')])
        lrs, y = Xn_to_Xy(lrs_dif, lrs_same)
        self.assertTrue(np.isnan(_devpavcalculator(lrs, PAVresult, y)))

        ### tests on ordinary data

        # test on dummy data. This PAV-transform is parallel to the identity line
        lrs_same = (1, 10**3)
        lrs_dif = (0.1, 10)
        PAVresult = np.array([0, 1, 1, float('inf')])
        lrs, y = Xn_to_Xy(lrs_dif, lrs_same)
        self.assertAlmostEqual(_devpavcalculator(lrs, PAVresult, y), 0.5)

        # test on dummy data 2, this PAV-transform crosses the identity line
        lrs_same = (0.1, 100, 10**3)
        lrs_dif = (10**-3, 10**-2, 10)
        fakePAVresult = np.array([0, 10**-3, 10**2, 10**-2, 10**2, float('inf')])
        lrs, y = Xn_to_Xy(lrs_dif, lrs_same)
        self.assertAlmostEqual(
            _devpavcalculator(lrs, fakePAVresult, y), (1 + 2 * (0.5 * 2 * 1 - 0.5 * 1 * 1) + 0.5) / 4
        )

        # test on dummy data 3, this PAV-transform is finite
        lrs_same = (0.1, 100)
        lrs_dif = (10**-2, 10)
        fakePAVresult = np.array([10**-3, 10**2, 10**-2, 10**2])
        lrs, y = Xn_to_Xy(lrs_dif, lrs_same)
        self.assertAlmostEqual(
            _devpavcalculator(lrs, fakePAVresult, y), (1 + 2 * (0.5 * 2 * 1 - 0.5 * 1 * 1) + 0.5) / 4
        )

    def test_calcsurface(self):
        # tests for the _calcsurface function

        # the line segment is parallel to the identity line
        c1 = (4, 1)
        c2 = (10, 7)
        self.assertAlmostEqual(_calcsurface(c1, c2), 18)

        # 2nd possibility (situation 1 of 2 in code below, the intersection with the identity line is within the line
        # segment, y1 < x1)
        c1 = (-1, -2)
        c2 = (0, 3)
        self.assertAlmostEqual(_calcsurface(c1, c2), 1.25)

        # 3rd possibility (situation 2 of 2 in code below, the intersection with the identity line is within the line
        # segment, y1 >= x1)
        c1 = (0, 3)
        c2 = (10, 4)
        self.assertAlmostEqual(_calcsurface(c1, c2), 25)

        # 5th possibility (situation 1 of 4 in code below, both coordinates are below the identity line, intersection
        # with identity line on left hand side)
        c1 = (-1, -2)
        c2 = (0, -1.5)
        self.assertAlmostEqual(_calcsurface(c1, c2), 1.25)

        # 6th possibility (situation 2 van 4 in code below, both coordinates are above the identity line, intersection
        # with identity line on right hand side)
        c1 = (1, 2)
        c2 = (1.5, 2)
        self.assertAlmostEqual(_calcsurface(c1, c2), 0.375)

        # 7th possibility (situation 2 of 4 in code below, both coordinates are above the identity line, intersection
        # with identity line on left hand side)
        c1 = (1, 2)
        c2 = (2, 4)
        self.assertAlmostEqual(_calcsurface(c1, c2), 1.5)

        # 8th possibility (situation 3 of 4 in code below, both coordinates are below the identity line, intersection
        # with identity line on right hand side)
        c1 = (-1, -2)
        c2 = (0, -0.5)
        self.assertAlmostEqual(_calcsurface(c1, c2), 0.75)

        # test with negative slope
        c1 = (1, 4)
        c2 = (2, 2)
        # self.assertEqual(_calcsurface(c1, c2), None)
        with self.assertRaises(ValueError):
            _calcsurface(c1, c2)
