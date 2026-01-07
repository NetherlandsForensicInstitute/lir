import logging
import math
import warnings
from collections.abc import Callable
from typing import Any, Self

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import KernelDensity

from lir.util import Xy_to_Xn, check_misleading_finite, ln_to_log10


LOG = logging.getLogger(__name__)


def compensate_and_remove_neginf_inf(
    log_odds: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    for Gaussian and KDE-calibrator fitting: remove negInf, Inf and compensate
    """
    X_finite = np.isfinite(log_odds).flatten()
    el_H1 = np.logical_and(X_finite, y == 1)
    el_H2 = np.logical_and(X_finite, y == 0)
    n_H1 = np.sum(y)
    numerator = np.sum(el_H1) / n_H1
    denominator = np.sum(el_H2) / (len(y) - n_H1)
    y = y[X_finite]
    log_odds = log_odds[X_finite]
    return log_odds, y, numerator, denominator


def parse_bandwidth(
    bandwidth: Callable | str | float | tuple[float, float] | None,
) -> Callable[[Any, Any], tuple[float, float]]:
    """
    Returns bandwidth as a tuple of two (optional) floats.
    Extrapolates a single bandwidth.

    :param bandwidth: provided bandwidth
    :return: bandwidth used for kde0, bandwidth used for kde1
    """
    match bandwidth:
        case None:
            raise ValueError('Missing `bandwidth` argument for KDE')

        case Callable():  #  type: ignore[misc]
            return bandwidth

        # string of specific supported bandwidth function
        case str():
            if bandwidth == 'silverman':
                return KDECalibrator.bandwidth_silverman

            # The given bandwidth method is not supported
            raise ValueError(f'Invalid input for bandwidth: {bandwidth}')

        # tuple or list
        case [int() | float() as bandwidth_0, int() | float() as bandwidth_1]:
            # Lambda function casting input to a tuple of the bandwidth ranges
            return lambda X, y: (bandwidth_0, bandwidth_1)

        case float() | int():
            # Lambda function casting input to a tuple of the bandwidth ranges
            return lambda X, y: (0 + bandwidth, bandwidth)

        case _:
            raise ValueError(f'Invalid `bandwidth` type: {type(bandwidth)} (value={bandwidth!r})')


class KDECalibrator(BaseEstimator, TransformerMixin):
    """
    Calculates a likelihood ratio of a score value, provided it is from one of
    two distributions. Uses kernel density estimation (KDE) for interpolation.
    """

    def __init__(self, bandwidth: Callable | str | float | tuple[float, float] | None = None):
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
        self.bandwidth: Callable = parse_bandwidth(bandwidth)
        self._kde0: KernelDensity | None = None
        self._kde1: KernelDensity | None = None
        self.numerator: float | None = None
        self.denominator: float | None = None

    @staticmethod
    def bandwidth_silverman(X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        """
        Estimates the optimal bandwidth parameter using Silverman's rule of
        thumb.
        """
        assert len(X) > 0 and len(y) > 0

        bandwidth = []
        for label in np.unique(y):
            values = X[y == label]
            std = np.std(values)
            if std == 0:
                # can happen eg if std(values) = 0
                warnings.warn(
                    'silverman bandwidth cannot be calculated if standard deviation is 0',
                    RuntimeWarning,
                    stacklevel=2,
                )
                LOG.info('found a silverman bandwidth of 0 (using dummy value)')
                std = 1

            v = math.pow(std, 5) / len(values) * 4.0 / 3
            bandwidth.append(math.pow(v, 0.2))

        if len(bandwidth) != 2:
            raise ValueError(f'expected 2 classes; found: {len(bandwidth)} classes: {y}')

        return bandwidth[0], bandwidth[1]

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        # check if we have matching dimensions
        if np.prod(X.shape) != len(y):
            raise ValueError(f'invalid shape: expected: ({len(y)},) or ({len(y)}, 1); found: {X.shape}')

        # make sure we have a 2d array of one column
        X = X.reshape(-1, 1)

        # check if data is sane
        check_misleading_finite(X, y)

        # KDE needs finite scale. Inf and negInf are treated as point masses at the extremes.
        # Remove them from data for KDE and calculate fraction data that is left.
        # LRs in finite range will be corrected for fractions in transform function
        X, y, self.numerator, self.denominator = compensate_and_remove_neginf_inf(X, y)
        X0, X1 = Xy_to_Xn(X, y)

        bandwidth0, bandwidth1 = self.bandwidth(X, y)
        self._kde0 = KernelDensity(kernel='gaussian', bandwidth=bandwidth0).fit(X0)
        self._kde1 = KernelDensity(kernel='gaussian', bandwidth=bandwidth1).fit(X1)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Provide LLR's as output."""
        if self._kde0 is None or self._kde1 is None or self.numerator is None or self.denominator is None:
            raise ValueError('KDECalibrator.transform() called before fit')

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

        return LLRs_output.flatten()
