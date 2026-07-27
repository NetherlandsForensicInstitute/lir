import numpy as np

from lir import LLRData
from lir.algorithms.isotonic_regression import IsotonicCalibrator
from lir.metrics.base import llr_metric
from lir.util import Xy_to_Xn, logodds_to_odds


@llr_metric
def cllr(llr_data: LLRData, weights: tuple[float, float] = (1, 1)) -> float:
    """
    Calculate a log likelihood ratio cost (C_llr) for a series of log likelihood ratios.

    Nico Brümmer and Johan du Preez, Application-independent evaluation of speaker detection, In: Computer Speech and
    Language 20(2-3), 2006.

    Parameters
    ----------
    llr_data : LLRData
        LLRs and their metadata, wrapped in an `LLRData` object.
    weights : tuple[float, float], optional
        The relative weights of the classes.

    Returns
    -------
    float
        CLLR, the log likelihood ratio cost.
    """
    llrs, y = llr_data.llrs, llr_data.require_labels

    lrs = logodds_to_odds(llrs)

    # ignore errors:
    #   divide -> ignore divide by zero
    #   over -> ignore scalar overflow
    with np.errstate(divide='ignore', over='ignore'):
        lrs0, lrs1 = Xy_to_Xn(lrs, y)
        if (weights[0] > 0 and len(lrs0) == 0) or (weights[1] > 0 and len(lrs1) == 0):
            return np.nan

        cllr0 = weights[0] * np.mean(np.log2(1 + lrs0)) if weights[0] > 0 else 0
        cllr1 = weights[1] * np.mean(np.log2(1 + 1 / lrs1)) if weights[1] > 0 else 0
        return float((cllr0 + cllr1) / sum(weights))


@llr_metric
def cllr_min(llr_data: LLRData, weights: tuple[float, float] = (1, 1)) -> float:
    """
    Estimate the discriminative power from a collection of log likelihood ratios.

    Parameters
    ----------
    llr_data : LLRData
        LLRs and their metadata, wrapped in an `LLRData` object.
    weights : tuple[float, float], optional
        The relative weights of the classes.

    Returns
    -------
    float
        CLLR_min, a measure of discrimination.
    """
    if not np.all(np.unique(llr_data.require_labels) == [0, 1]):
        return np.nan

    cal = IsotonicCalibrator()
    llrmin = cal.fit_apply(llr_data)

    return cllr(llrmin, weights)  # type: ignore


@llr_metric
def cllr_cal(llr_data: LLRData, weights: tuple[float, float] = (1, 1)) -> float:
    """
    Calculate the difference between the C_llr before and after isotonic calibration.

    Parameters
    ----------
    llr_data : LLRData
        LLRs and their metadata, wrapped in an `LLRData` object.
    weights : tuple[float, float], optional
        The relative weights of the classes.

    Returns
    -------
    float
        CLLR_cal, the difference after isotonic calibration.
    """
    cllr_min_val = cllr_min(llr_data, weights)  # type: ignore
    cllr_val = cllr(llr_data, weights)  # type: ignore

    return cllr_val - cllr_min_val
