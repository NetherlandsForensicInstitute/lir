import numpy as np
import pytest

from lir import metrics
from lir.data.models import LLRData
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
    llr_data = LLRData(features=llrs, hypothesis_labels=labels)
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
    llr_data = LLRData(features=llrs, hypothesis_labels=labels)
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
    llr_data = LLRData(features=llrs, hypothesis_labels=labels)
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
    llr_data = LLRData(features=llrs, hypothesis_labels=labels)
    assert np.isnan(metrics.cllr(llr_data))

    if np.all(np.isfinite(llrs)):  # this condition should be removed ?! --> see issue #301
        assert np.isnan(metrics.cllr_min(llr_data))
