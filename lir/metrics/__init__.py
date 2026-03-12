import numpy as np

from lir.algorithms.isotonic_regression import IsotonicCalibrator
from lir.data.models import LLRData
from lir.util import Xy_to_Xn, logodds_to_odds


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

    return cllr(llrmin, weights)


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
    cllr_min_val = cllr_min(llr_data, weights)
    cllr_val = cllr(llr_data, weights)

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
