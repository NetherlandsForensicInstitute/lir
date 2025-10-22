import logging
import math
import warnings
from typing import Sized, Callable, Union, Tuple, Optional

import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.neighbors import KernelDensity

from lir.util import ln_to_log10, Xy_to_Xn, check_misleading_finite

LOG = logging.getLogger(__name__)


def compensate_and_remove_neginf_inf(
    log_odds: np.ndarray, y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    for Gaussian and KDE-calibrator fitting: remove negInf, Inf and compensate
    """
    X_finite = np.isfinite(log_odds)
    el_H1 = np.logical_and(X_finite, y == 1)
    el_H2 = np.logical_and(X_finite, y == 0)
    n_H1 = np.sum(y)
    numerator = np.sum(el_H1) / n_H1
    denominator = np.sum(el_H2) / (len(y) - n_H1)
    y = y[X_finite]
    log_odds = log_odds[X_finite]
    return log_odds, y, numerator, denominator


class KDECalibrator(BaseEstimator, TransformerMixin):
    """
    Calculates a likelihood ratio of a score value, provided it is from one of
    two distributions. Uses kernel density estimation (KDE) for interpolation.
    """

    def __init__(
        self, bandwidth: Union[Callable, str, float, Tuple[float, float]] = None
    ):
        """

        :param bandwidth:
            * If bandwidth has a float value, this value is used as the bandwidth for both distributions.
            * If bandwidth is a tuple, it should contain two floating point values: the bandwidth for the distribution
              of the classes with labels 0 and 1, respectively.
            * If bandwidth has the str value "silverman", Silverman's rule of thumb is used as the bandwidth for both
              distributions separately.
            * If bandwidth is callable, it should accept two arguments, `X` and `y`, and return a tuple of two values
              which are the bandwidths for the two distributions.
        """
        self.bandwidth: Callable = self._parse_bandwidth(bandwidth)
        self._kde0: Optional[KernelDensity] = None
        self._kde1: Optional[KernelDensity] = None
        self.numerator, self.denominator = None, None

    @staticmethod
    def bandwidth_silverman(X, y):
        """
        Estimates the optimal bandwidth parameter using Silverman's rule of
        thumb.
        """
        assert len(X) > 0

        bandwidth = []
        for label in np.unique(y):
            values = X[y == label]
            std = np.std(values)
            if std == 0:
                # can happen eg if std(values) = 0
                warnings.warn(
                    "silverman bandwidth cannot be calculated if standard deviation is 0",
                    RuntimeWarning,
                )
                LOG.info("found a silverman bandwidth of 0 (using dummy value)")
                std = 1

            v = math.pow(std, 5) / len(values) * 4.0 / 3
            bandwidth.append(math.pow(v, 0.2))

        return bandwidth

    @staticmethod
    def bandwidth_scott(X, y):
        """
        Not implemented.
        """
        raise

    def fit(self, X, y):
        # check if data is sane
        check_misleading_finite(X, y)

        # KDE needs finite scale. Inf and negInf are treated as point masses at the extremes.
        # Remove them from data for KDE and calculate fraction data that is left.
        # LRs in finite range will be corrected for fractions in transform function
        X, y, self.numerator, self.denominator = compensate_and_remove_neginf_inf(X, y)
        X0, X1 = Xy_to_Xn(X, y)
        X0 = X0.reshape(-1, 1)
        X1 = X1.reshape(-1, 1)

        bandwidth0, bandwidth1 = self.bandwidth(X, y)
        self._kde0 = KernelDensity(kernel="gaussian", bandwidth=bandwidth0).fit(X0)
        self._kde1 = KernelDensity(kernel="gaussian", bandwidth=bandwidth1).fit(X1)
        return self

    def transform(self, X):
        """Provide LLR's as output."""
        assert self._kde0 is not None, "KDECalibrator.transform() called before fit"
        self.p0 = np.empty(np.shape(X))
        self.p1 = np.empty(np.shape(X))
        # initiate LRs_output
        LLRs_output = np.empty(np.shape(X))

        # get inf and neginf
        wh_inf = np.isposinf(X)
        wh_neginf = np.isneginf(X)

        # assign hard values for extremes
        LLRs_output[wh_inf] = np.inf
        LLRs_output[wh_neginf] = -np.inf
        self.p0[wh_inf] = 0
        self.p1[wh_inf] = 1
        self.p0[wh_neginf] = 1
        self.p1[wh_neginf] = 0

        # get elements that are not inf or neginf
        el = np.isfinite(X)
        X = X[el]

        # perform KDE as usual
        X = X.reshape(-1, 1)
        ln_H1 = self._kde1.score_samples(X)
        ln_H2 = self._kde0.score_samples(X)
        ln_dif = ln_H1 - ln_H2
        log10_dif = ln_to_log10(ln_dif)

        # calculate p0 and p1's (redundant?)
        self.p0[el] = self.denominator * np.exp(ln_H2)
        self.p1[el] = self.numerator * np.exp(ln_H1)

        # apply correction for fraction of negInf and Inf data
        log10_compensator = np.log10(self.numerator / self.denominator)
        LLRs_output[el] = log10_compensator + log10_dif

        return LLRs_output

    @staticmethod
    def _parse_bandwidth(
        bandwidth: Union[Callable, float, Tuple[float, float]],
    ) -> Callable:
        """
        Returns bandwidth as a tuple of two (optional) floats.
        Extrapolates a single bandwidth
        :param bandwidth: provided bandwidth
        :return: bandwidth used for kde0, bandwidth used for kde1
        """
        if bandwidth is None:
            raise ValueError("missing bandwidth argument for KDE")
        elif callable(bandwidth):
            return bandwidth
        elif bandwidth == "silverman":
            return KDECalibrator.bandwidth_silverman
        elif bandwidth == "scott":
            return KDECalibrator.bandwidth_scott
        elif isinstance(bandwidth, str):
            raise ValueError(f"invalid input for bandwidth: {bandwidth}")
        elif isinstance(bandwidth, Sized):
            assert (
                len(bandwidth) == 2
            ), f"bandwidth should have two elements; found {len(bandwidth)}; bandwidth = {bandwidth}"
            return lambda X, y: bandwidth
        else:
            return lambda X, y: (0 + bandwidth, bandwidth)
