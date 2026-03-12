import functools
import logging
import math
import warnings
from collections.abc import Callable
from typing import Self

import numpy as np
from sklearn.neighbors import KernelDensity

from lir import Transformer
from lir.data.models import FeatureData, InstanceData, LLRData
from lir.util import check_type, ln_to_log10


LOG = logging.getLogger(__name__)


def _compensate_and_remove_neginf_inf(data: LLRData) -> tuple[LLRData, float, float]:
    """
    Remove infinite log-odds values and compute compensation factors.

    Parameters
    ----------
    data : LLRData
        The LLRs.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, float, float]
        Finite log-odds, corresponding labels, numerator compensator, and denominator compensator.
    """
    X_finite = np.isfinite(data.llrs)
    el_H1 = np.logical_and(X_finite, data.labels == 1)
    el_H2 = np.logical_and(X_finite, data.labels == 0)
    n_H1 = np.sum(data.require_labels)
    numerator = np.sum(el_H1) / n_H1
    denominator = np.sum(el_H2) / (len(data) - n_H1)

    return data[X_finite], numerator, denominator


def _fixed_bandwidth(instances: FeatureData, *, bandwidth_0: float, bandwidth_1: float) -> tuple[float, float]:
    """
    Return a fixed bandwidth pair.

    Parameters
    ----------
    instances : FeatureData
        Unused input data placeholder.
    bandwidth_0 : float
        Bandwidth for class 0.
    bandwidth_1 : float
        Bandwidth for class 1.

    Returns
    -------
    tuple[float, float]
        Fixed bandwidths for class 0 and class 1.
    """
    return bandwidth_0, bandwidth_1


BandwidthFunction = Callable[[FeatureData], tuple[float, float]]


def parse_bandwidth(
    bandwidth: Callable | str | float | tuple[float, float] | None,
) -> BandwidthFunction:
    """
    Parse and return the corresponding bandwidth strategy based on input type.

    Returns bandwidth as a tuple of two (optional) floats.
    Extrapolates a single bandwidth.

    Parameters
    ----------
    bandwidth : Callable | str | float | tuple[float, float] | None
        Bandwidth specification.

    Returns
    -------
    Callable[[Any, Any], tuple[float, float]]
        Callable that computes bandwidths for class 0 and class 1.
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
            # Use functools.partial (picklable) instead of a lambda closure
            return functools.partial(_fixed_bandwidth, bandwidth_0=bandwidth_0, bandwidth_1=bandwidth_1)

        case float() | int():
            # Use functools.partial (picklable) instead of a lambda closure
            return functools.partial(_fixed_bandwidth, bandwidth_0=float(bandwidth), bandwidth_1=float(bandwidth))

        case _:
            raise ValueError(f'Invalid `bandwidth` type: {type(bandwidth)} (value={bandwidth!r})')


class KDECalibrator(Transformer):
    """
    Calculate LR from a score, belonging to one of two distributions using KDE.

    Calculates a likelihood ratio of a score value, provided it is from one of
    two distributions. Uses kernel density estimation (KDE) for interpolation.

    Parameters
    ----------
    bandwidth : Callable | str | float | tuple[float, float] | None, optional
        Bandwidth specification for KDE.
    """

    def __init__(self, bandwidth: BandwidthFunction | str | float | tuple[float, float] | None = None):
        """
        Initialize a new KDECalibrator instance.

        Parameters
        ----------
        bandwidth : Callable | str | float | tuple[float, float] | None, optional
                Bandwidth specification for KDE.
        """
        self.bandwidth = parse_bandwidth(bandwidth)
        self._kde0: KernelDensity | None = None
        self._kde1: KernelDensity | None = None
        self.numerator: float | None = None
        self.denominator: float | None = None

    @staticmethod
    def bandwidth_silverman(instances: FeatureData) -> tuple[float, float]:
        r"""
        Estimate bandwidths using Silverman's rule of thumb.

        This method calculates the bandwidth for both hypotheses separately. It uses two parameters:
        - the standard deviation of the training data
        - the size of the training data

        If the instances have values for `source_ids`, the size is calculated as the number of distinct source ids.
        Otherwise, the size is the number of instances.

        The bandwidth is calculated as follows, where σ is the standard deviation and n is the size.

        .. math::

            \left( \frac{4 \sigma^5}{3 n} \right)^{\frac{1}{5}}

        Parameters
        ----------
        instances : FeatureData
            Feature data to be used for bandwidth calculation.

        Returns
        -------
        tuple[float, float]
            Bandwidth for class 0 and class 1.
        """
        bandwidth = []
        for label in np.unique(instances.require_both_labels):
            features = instances.features[instances.labels == label]
            std = np.std(features)
            size = len(np.unique(instances.source_ids)) if instances.source_ids is not None else features.shape[0]

            if std == 0:
                # can happen eg if std(values) = 0
                warnings.warn(
                    'silverman bandwidth cannot be calculated if standard deviation is 0',
                    RuntimeWarning,
                    stacklevel=2,
                )
                LOG.info('found a silverman bandwidth of 0 (using dummy value)')
                std = 1

            v = math.pow(std, 5) / size * 4.0 / 3
            bandwidth.append(math.pow(v, 0.2))
            LOG.debug(
                f'calculated KDE bandwidth: {bandwidth[-1]}; used silverman with parameters: size={size}; std={std}'
            )

        return bandwidth[0], bandwidth[1]

    def fit(self, instances: InstanceData) -> Self:
        """
        Fit the KDE model on the data.

        Parameters
        ----------
        instances : InstanceData
            Training instances.

        Returns
        -------
        Self
            Fitted calibrator.
        """
        instances = check_type(FeatureData, instances)
        instances = instances.replace_as(LLRData)

        # check if data is sane
        instances.check_misleading_finite()

        # KDE needs finite scale. Inf and negInf are treated as point masses at the extremes.
        # Remove them from data for KDE and calculate fraction data that is left.
        # LRs in finite range will be corrected for fractions in the apply function
        data, self.numerator, self.denominator = _compensate_and_remove_neginf_inf(instances)

        bandwidth0, bandwidth1 = self.bandwidth(data)
        self._kde0 = KernelDensity(kernel='gaussian', bandwidth=bandwidth0).fit(data[data.labels == 0].features)
        self._kde1 = KernelDensity(kernel='gaussian', bandwidth=bandwidth1).fit(data[data.labels == 1].features)
        return self

    def apply(self, instances: InstanceData) -> LLRData:
        """
        Provide calibrated LLRs as output.

        Parameters
        ----------
        instances : InstanceData
            Instances to calibrate.

        Returns
        -------
        LLRData
            Calibrated log-likelihood-ratio data.
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
        return instances.replace(
            features=llrs_output,
            probabilities=probabilities,
        )
