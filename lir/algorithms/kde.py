import logging
import math
import warnings
from collections.abc import Callable
from typing import Any, Self

import numpy as np
from sklearn.neighbors import KernelDensity

from lir import Transformer
from lir.data.models import FeatureData, InstanceData, LLRData
from lir.util import Xy_to_Xn, check_type, ln_to_log10


LOG = logging.getLogger(__name__)


def compensate_and_remove_neginf_inf(
    log_odds: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """For Gaussian and KDE-calibrator fitting: remove negInf, Inf and compensate.

    :param log_odds: n * 1 np.array of log-odds
    :param y: n * 1 np.array of labels (Booleans).

    :returns: log_odds (with negInf and Inf removed), y (with negInf and Inf removed),
                numerator compensator, denominator compensator
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


class KDECalibrator(Transformer):
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

        :param X: n * 1 np.array of scores
        :param y: n * 1 np.array of labels (Booleans).
        :returns: bandwidth for class 0, bandwidth for class 1
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

    def fit(self, instances: InstanceData) -> Self:
        instances = check_type(FeatureData, instances)
        instances = instances.replace_as(LLRData)

        # check if data is sane
        instances.check_misleading_finite()

        # KDE needs finite scale. Inf and negInf are treated as point masses at the extremes.
        # Remove them from data for KDE and calculate fraction data that is left.
        # LRs in finite range will be corrected for fractions in the apply function
        X, y, self.numerator, self.denominator = compensate_and_remove_neginf_inf(
            instances.llrs.reshape(-1, 1), instances.require_labels
        )
        X0, X1 = Xy_to_Xn(X, y)

        bandwidth0, bandwidth1 = self.bandwidth(X, y)
        self._kde0 = KernelDensity(kernel='gaussian', bandwidth=bandwidth0).fit(X0)
        self._kde1 = KernelDensity(kernel='gaussian', bandwidth=bandwidth1).fit(X1)
        return self

    def apply(self, instances: InstanceData) -> LLRData:
        """Provide LLR's as output.

        :param instances: InstanceData to apply the calibrator to.
        :returns: LLRData with calibrated log-likelihood ratios.
        """
        if self._kde0 is None or self._kde1 is None or self.numerator is None or self.denominator is None:
            raise ValueError('KDECalibrator.apply() called before fit')

        instances = check_type(FeatureData, instances)
        instances = instances.replace_as(LLRData)

        # initiate LRs_output
        llrs_output = np.empty(instances.llrs.shape)
        p0 = np.empty(instances.llrs.shape)
        p1 = np.empty(instances.llrs.shape)

        # get inf and neginf
        wh_inf = np.isposinf(instances.llrs)
        wh_neginf = np.isneginf(instances.llrs)

        # assign hard values for extremes
        llrs_output[wh_inf] = np.inf
        llrs_output[wh_neginf] = -np.inf
        p0[wh_inf] = 0
        p1[wh_inf] = 1
        p0[wh_neginf] = 1
        p1[wh_neginf] = 0

        # get elements that are not inf or neginf
        finite_llrs_index = np.isfinite(instances.llrs)
        finite_llrs = instances.llrs[finite_llrs_index].reshape(-1, 1)

        # perform KDE as usual
        ln_H1 = self._kde1.score_samples(finite_llrs)
        ln_H2 = self._kde0.score_samples(finite_llrs)
        ln_dif = ln_H1 - ln_H2
        log10_dif = ln_to_log10(ln_dif)

        # calculate p0 and p1's (redundant?)
        p0[finite_llrs_index] = self.denominator * np.exp(ln_H2)
        p1[finite_llrs_index] = self.numerator * np.exp(ln_H1)

        # apply correction for fraction of negInf and Inf data
        log10_compensator = np.log10(self.numerator / self.denominator)
        llrs_output[finite_llrs_index] = log10_compensator + log10_dif

        probabilities = np.stack([p0, p1], axis=1)
        return instances.replace(features=llrs_output, probabilities=probabilities)
