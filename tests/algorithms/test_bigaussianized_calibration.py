import numpy as np

from lir import LLRData, metrics
from lir.algorithms.bigaussianized_calibration import BigaussianCalibrator
from lir.algorithms.logistic_regression import LogitCalibrator
from lir.util import check_type


def _ln_to_log10[Type: np.ndarray | float](values: Type) -> Type:
    return np.log10(np.exp(values))


def test_cllr_to_variance():
    rng = np.random.default_rng(seed=42)
    mean = _ln_to_log10(4.5)
    std = _ln_to_log10(3)
    h1 = rng.normal(loc=mean, scale=std, size=1000)
    h2 = rng.normal(loc=-mean, scale=std, size=1000)

    llrs = LLRData(features=np.concatenate([h1, h2]), hypothesis=np.concatenate([np.ones(1000, dtype=int), np.zeros(1000, dtype=int)]))
    np.testing.assert_almost_equal(np.mean(llrs.llrs), 0, decimal=1)
    np.testing.assert_almost_equal(np.mean(llrs.llrs[llrs.hypothesis==0]), -mean, decimal=1)
    np.testing.assert_almost_equal(np.mean(llrs.llrs[llrs.hypothesis==1]), mean, decimal=1)

    calibrated_llrs = check_type(LLRData, LogitCalibrator(random_state=42).fit_apply(llrs))
    np.testing.assert_almost_equal(np.mean(calibrated_llrs.llrs), 0, decimal=1)
    np.testing.assert_almost_equal(np.mean(calibrated_llrs.llrs[calibrated_llrs.hypothesis==0]), -mean, decimal=1)
    np.testing.assert_almost_equal(np.mean(calibrated_llrs.llrs[calibrated_llrs.hypothesis==1]), mean, decimal=1)

    derived_variance = BigaussianCalibrator._cllr_to_variance(metrics.cllr(llrs))
    np.testing.assert_almost_equal(derived_variance, std*std, decimal=1)


def test_paper_fig2a():
    rng = np.random.default_rng(seed=42)
    mean = _ln_to_log10(4.5)
    std = _ln_to_log10(3)
    h1 = rng.normal(loc=mean, scale=std, size=1000)
    h2 = rng.normal(loc=-mean, scale=std, size=1000)

    llrs = LLRData(features=np.concatenate([h1, h2]), hypothesis=np.concatenate([np.ones(1000, dtype=int), np.zeros(1000, dtype=int)]))
    calibrated_llrs = check_type(LLRData, BigaussianCalibrator(LogitCalibrator()).fit_apply(llrs))
    np.testing.assert_almost_equal(np.log(np.mean(calibrated_llrs.llrs)), 0)
