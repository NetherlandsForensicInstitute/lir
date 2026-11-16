import warnings
from typing import Any, Protocol

import numpy as np
import sklearn

from lir.algorithms.isotonic_regression import IsotonicCalibrator
from lir.data.models import LLRData
from lir.util import Xy_to_Xn, logodds_to_odds


class MetricType(Protocol):
    """Type definition for metric calculations."""

    def __call__(self, llrdata: LLRData) -> float | None:
        """
        Call the metric.

        Parameters
        ----------
        llrdata : LLRData
            The input data.
        """
        ...


def _resolve_h1_weight(h1_weight: float, **kwargs: Any) -> float:
    """
    Resolve the weight of hypothesis H1, accepting the deprecated ``weights`` argument for backward compatibility.

    Parameters
    ----------
    h1_weight : float
        The weight of hypothesis H1, as a fraction of the total weight.
    kwargs : dict[str, Any]
        Additional keyword arguments. The deprecated ``weights`` tuple is taken from here, and any other
        argument is rejected.

    Returns
    -------
    float
        The resolved weight of hypothesis H1, as a fraction of the total weight.
    """
    if 'weights' in kwargs:
        weights = kwargs.pop('weights')
        if not isinstance(weights, (list, tuple)) or len(weights) != 2:
            raise ValueError(f'expected `weights` to be a sequence of two numbers; found: {weights!r}')
        h2_weight, h1_weight = float(weights[0]), float(weights[1])
        if h2_weight + h1_weight <= 0:
            raise ValueError(f'expected the sum of `weights` to be positive; found: {weights!r}')
        warnings.warn(
            '`weights` is deprecated; use `h1_weight` instead, where `h1_weight` is the weight of H1 as a fraction '
            'of the total weight, i.e. `h1_weight = weights[1] / sum(weights)`.',
            DeprecationWarning,
            stacklevel=3,
        )
        h1_weight = h1_weight / (h1_weight + h2_weight)
    if kwargs:
        raise TypeError(f'unexpected keyword argument(s): {", ".join(kwargs)!r}')
    return h1_weight


def cllr(llr_data: LLRData, h1_weight: float = 0.5, **kwargs: Any) -> float:
    """
    Calculate a log likelihood ratio cost (C_llr) for a series of log likelihood ratios.

    Nico Brümmer and Johan du Preez, Application-independent evaluation of speaker detection, In: Computer Speech and
    Language 20(2-3), 2006.

    Parameters
    ----------
    llr_data : LLRData
        LLRs and their metadata, wrapped in an `LLRData` object.
    h1_weight : float, optional
        The relative weight of hypothesis H1, as a fraction of 1; H2 gets the remaining weight ``1 - h1_weight``.
        Defaults to 0.5, which weighs both hypotheses equally.
    kwargs : dict[str, Any], optional
        For backward compatibility only: the deprecated ``weights`` tuple, where ``weights[1] / sum(weights)``
        gives the equivalent value of ``h1_weight``. Emits a :class:`DeprecationWarning`.

    Returns
    -------
    float
        CLLR, the log likelihood ratio cost.

    Examples
    --------
    The weight can also be set in the YAML configuration of an experiment:

    .. code-block:: yaml

        output:
          - method: metrics
            columns:
              - cllr
              - name: cllr_weighted
                method: cllr
                h1_weight: 0.9
    """
    h1_weight = _resolve_h1_weight(h1_weight, **kwargs)
    if not 0 <= h1_weight <= 1:
        raise ValueError(f'expected `h1_weight` to be between 0 and 1; found: {h1_weight}')
    h2_weight = 1 - h1_weight

    llrs, y = llr_data.llrs, llr_data.require_labels

    lrs = logodds_to_odds(llrs)

    # ignore errors:
    #   divide -> ignore divide by zero
    #   over -> ignore scalar overflow
    with np.errstate(divide='ignore', over='ignore'):
        lrs0, lrs1 = Xy_to_Xn(lrs, y)
        if (h2_weight > 0 and len(lrs0) == 0) or (h1_weight > 0 and len(lrs1) == 0):
            return np.nan

        cllr0 = h2_weight * np.mean(np.log2(1 + lrs0)) if h2_weight > 0 else 0
        cllr1 = h1_weight * np.mean(np.log2(1 + 1 / lrs1)) if h1_weight > 0 else 0
        return float(cllr0 + cllr1)


def cllr_min(llr_data: LLRData, h1_weight: float = 0.5, **kwargs: Any) -> float:
    """
    Estimate the discriminative power from a collection of log likelihood ratios.

    Parameters
    ----------
    llr_data : LLRData
        LLRs and their metadata, wrapped in an `LLRData` object.
    h1_weight : float, optional
        The relative weight of hypothesis H1, as a fraction of 1; H2 gets the remaining weight ``1 - h1_weight``.
        Defaults to 0.5, which weighs both hypotheses equally.

    Returns
    -------
    float
        CLLR_min, a measure of discrimination.

    Notes
    -----
    For backward compatibility, the deprecated ``weights`` tuple is accepted in place of ``h1_weight``; see
    :func:`~lir.metrics.cllr`.
    """
    h1_weight = _resolve_h1_weight(h1_weight, **kwargs)
    if not np.all(np.unique(llr_data.require_labels) == [0, 1]):
        return np.nan

    cal = IsotonicCalibrator()
    llrmin = cal.fit_apply(llr_data)

    return cllr(llrmin, h1_weight)


def cllr_cal(llr_data: LLRData, h1_weight: float = 0.5, **kwargs: Any) -> float:
    """
    Calculate the difference between the C_llr before and after isotonic calibration.

    Parameters
    ----------
    llr_data : LLRData
        LLRs and their metadata, wrapped in an `LLRData` object.
    h1_weight : float, optional
        The relative weight of hypothesis H1, as a fraction of 1; H2 gets the remaining weight ``1 - h1_weight``.
        Defaults to 0.5, which weighs both hypotheses equally.

    Returns
    -------
    float
        CLLR_cal, the difference after isotonic calibration.

    Notes
    -----
    For backward compatibility, the deprecated ``weights`` tuple is accepted in place of ``h1_weight``; see
    :func:`~lir.metrics.cllr`.
    """
    h1_weight = _resolve_h1_weight(h1_weight, **kwargs)
    cllr_min_val = cllr_min(llr_data, h1_weight)
    cllr_val = cllr(llr_data, h1_weight)

    return cllr_val - cllr_min_val


def llr_upper_bound(llrs: LLRData) -> float | None:
    """
    Provide corresponding upper bound for provided LLR data.

    When an LLRData object contains an upper bound, return it. If not, return None.

    Parameters
    ----------
    llrs : LLRData
        LLRs and their metadata, wrapped in an `LLRData` object.

    Returns
    -------
    float | None
        The LLR upper bound, or `None`.
    """
    return llrs.llr_upper_bound


def llr_lower_bound(llrs: LLRData) -> float | None:
    """
    Provide corresponding lower bound for provided LLR data.

    When an LLRData object contains a lower bound, return it. If not, return None.

    Parameters
    ----------
    llrs : LLRData
        LLRs and their metadata, wrapped in an `LLRData` object.

    Returns
    -------
    float | None
        The LLR lower bound, or `None`.
    """
    return llrs.llr_lower_bound


def equal_error_rate(data: LLRData) -> float:
    """
    Calculate the Equal Error Rate (EER) from the ROC curve.

    The EER is the operating point where the True Positive Rate (TPR) equals
    the False Negative Rate (FNR).

    Parameters
    ----------
    data : LLRData
        LLRs and their metadata, wrapped in an `LLRData` object.

    Returns
    -------
    float
        The equal error rate.
    """
    hypothesis = data.require_labels
    fpr, tpr, threshold = sklearn.metrics.roc_curve(hypothesis, data.llrs)
    fnr = 1 - tpr

    return fpr[np.nanargmin(np.absolute(fnr - fpr))]


def auc(data: LLRData) -> float:
    """
    Calculate the Area Under the Curve (AUC) of the ROC curve.

    Parameters
    ----------
    data : LLRData
        LLRs and their metadata, wrapped in an `LLRData` object.

    Returns
    -------
    float
        The area under the ROC curve (AUC score).
    """
    return sklearn.metrics.roc_auc_score(data.hypothesis, data.llrs)
