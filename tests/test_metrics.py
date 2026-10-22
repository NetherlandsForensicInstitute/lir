import numpy as np
import pytest

from lir import metrics
from lir.data.models import LLRData
from lir.metrics.statistics import count, count_sources
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
    llr_data = LLRData(features=llrs, hypothesis=labels)
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
    llr_data = LLRData(features=llrs, hypothesis=labels)
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
    llr_data = LLRData(features=llrs, hypothesis=labels)
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
    llr_data = LLRData(features=llrs, hypothesis=labels)
    assert np.isnan(metrics.cllr(llr_data))

    if np.all(np.isfinite(llrs)):  # this condition should be removed ?! --> see issue #301
        assert np.isnan(metrics.cllr_min(llr_data))


def test_count():
    llrs, labels = Xn_to_Xy(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
    llr_data = LLRData(features=llrs, hypothesis=labels)
    assert count(llr_data) == 4


def test_count_sources():
    llrs, labels = Xn_to_Xy(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
    source_ids = np.array([1, 1, 2, 3])
    llr_data = LLRData(features=llrs, hypothesis=labels, source_ids=source_ids)
    assert count_sources(llr_data) == 3


def test_count_sources_paired():
    llrs, labels = Xn_to_Xy(np.array([1.0, 2.0]), np.array([3.0]))
    source_ids = np.array([[1, 2], [2, 3], [3, 1]])
    llr_data = LLRData(features=llrs, hypothesis=labels, source_ids=source_ids)
    assert count_sources(llr_data) == 3


def test_count_sources_missing():
    llrs, labels = Xn_to_Xy(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
    llr_data = LLRData(features=llrs, hypothesis=labels)
    with pytest.warns(UserWarning):
        assert np.isnan(count_sources(llr_data))


def make_llr_data(h1_llrs, h2_llrs) -> LLRData:
    """
    Build `LLRData` from per-hypothesis LLR values.

    H1 instances are labeled with hypothesis 1 and H2 instances with hypothesis 0, matching the
    convention that high LLR values support H1.
    """
    h1 = np.asarray(h1_llrs, dtype=float)
    h2 = np.asarray(h2_llrs, dtype=float)
    llrs = np.concatenate([h1, h2])
    hypothesis = np.concatenate([np.ones(len(h1)), np.zeros(len(h2))])
    return LLRData(features=llrs, hypothesis=hypothesis)


@pytest.mark.parametrize(
    'expected,h1_llrs,h2_llrs',
    [
        (1.0, [2, 1], [0, -1]),  # perfectly separated
        (0.0, [0, -1], [2, 1]),  # perfectly inverted
        (0.25, [0, 2], [1, 3]),  # one of four H1/H2 pairs is correctly ordered
        (0.5, [1, 2], [0, 3]),  # two of four H1/H2 pairs are correctly ordered
        (0.875, [1, 2], [0, 1]),  # tied pair at 1 counts as half
        (0.5, [1, 1], [1, 1]),  # all scores tied
    ],
)
def test_auc(expected: float, h1_llrs: list[float], h2_llrs: list[float]):
    llr_data = make_llr_data(h1_llrs, h2_llrs)
    assert metrics.auc(llr_data) == pytest.approx(expected)


def test_auc_is_bounded():
    rng = np.random.default_rng(42)
    llr_data = make_llr_data(rng.normal(size=50), rng.normal(size=50))
    assert 0.0 <= metrics.auc(llr_data) <= 1.0


def test_auc_inversion_symmetry():
    rng = np.random.default_rng(42)
    h1 = rng.normal(size=50)
    h2 = rng.normal(size=50)
    auc = metrics.auc(make_llr_data(h1, h2))
    auc_inverted = metrics.auc(make_llr_data(-h1, -h2))
    assert auc + auc_inverted == pytest.approx(1.0)


@pytest.mark.parametrize(
    'expected,h1_llrs,h2_llrs',
    [
        (0.0, [3, 2], [1, 0]),  # perfectly separated: TPR=1 at FPR=0
        (0.5, [0, 1], [0, 1]),  # symmetric: ROC crosses the anti-diagonal at (0.5, 0.5)
        (1.0, [1, 0], [3, 2]),  # perfectly inverted: FPR=FNR=1
        (0.25, [4, 3, 1], [2, 0, -1, -2]),  # ROC point closest to the anti-diagonal is at FPR=0.25
    ],
)
def test_equal_error_rate(expected: float, h1_llrs: list[float], h2_llrs: list[float]):
    llr_data = make_llr_data(h1_llrs, h2_llrs)
    assert metrics.equal_error_rate(llr_data) == pytest.approx(expected)


def test_equal_error_rate_is_bounded():
    rng = np.random.default_rng(42)
    llr_data = make_llr_data(rng.normal(size=100), rng.normal(size=100))
    assert 0.0 <= metrics.equal_error_rate(llr_data) <= 1.0


def test_equal_error_rate_is_invariant_to_monotonic_transforms():
    h1 = [4.0, 3.0, 2.5, 1.0, 0.5]
    h2 = [3.0, 1.5, 0.0, -1.0, -2.0]
    base = metrics.equal_error_rate(make_llr_data(h1, h2))
    scaled = metrics.equal_error_rate(make_llr_data([2 * x for x in h1], [2 * x for x in h2]))
    shifted = metrics.equal_error_rate(make_llr_data([x + 5 for x in h1], [x + 5 for x in h2]))
    assert base == scaled == shifted


def test_auc_and_equal_error_rate_ignore_intervals():
    h1, h2 = [4.0, 3.0], [2.0, 1.0]
    llr_data = make_llr_data(h1, h2)
    llr_values = llr_data.llrs
    features = np.column_stack([llr_values, llr_values - 1, llr_values + 1])
    llr_data_with_intervals = LLRData(features=features, hypothesis=llr_data.hypothesis)
    assert metrics.auc(llr_data_with_intervals) == metrics.auc(llr_data)
    assert metrics.equal_error_rate(llr_data_with_intervals) == metrics.equal_error_rate(llr_data)
