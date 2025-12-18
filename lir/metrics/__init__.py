import numpy as np

from lir.algorithms.isotonic_regression import IsotonicCalibrator
from lir.data.models import LLRData
from lir.util import Xy_to_Xn, logodds_to_odds


def cllr(llr_data: LLRData, weights: tuple[float, float] = (1, 1)) -> float:
    """
    Calculates a log likelihood ratio cost (C_llr) for a series of log likelihood
    ratios.

    Nico Brümmer and Johan du Preez, Application-independent evaluation of speaker detection, In: Computer Speech and
    Language 20(2-3), 2006.

    :param llrs: a numpy array of LLRs
    :param y: a numpy array of labels (0 or 1)
    :param weights: the relative weights of the classes
    :return: CLLR, the log likelihood ratio cost
    """

    # ignore errors:
    #   divide -> ignore divide by zero
    #   over -> ignore scalar overflow

    llrs, y = llr_data.llrs, llr_data.labels
    if y is None:
        raise ValueError('Labels are required to compute C_llr')

    lrs = logodds_to_odds(llrs)
    with np.errstate(divide='ignore', over='ignore'):
        lrs0, lrs1 = Xy_to_Xn(lrs, y)
        cllr0 = weights[0] * np.mean(np.log2(1 + lrs0)) if weights[0] > 0 else 0
        cllr1 = weights[1] * np.mean(np.log2(1 + 1 / lrs1)) if weights[1] > 0 else 0
        return float((cllr0 + cllr1) / sum(weights))


def cllr_min(llr_data: LLRData, weights: tuple[float, float] = (1, 1)) -> float:
    """
    Estimates the discriminative power from a collection of log likelihood ratios.

    :param llrs: a numpy array of LLRs
    :param y: a numpy array of labels (0 or 1)
    :param weights: the relative weights of the classes
    :return: CLLR_min, a measure of discrimination
    """
    llrs, y = llr_data.llrs, llr_data.labels
    if y is None:
        raise ValueError('Labels are required to compute C_llr')

    cal = IsotonicCalibrator()
    llrmin = cal.fit_transform(llrs, y)
    return cllr(LLRData(features=llrmin, labels=y), weights)


def cllr_cal(llr_data: LLRData, weights: tuple[float, float] = (1, 1)) -> float:
    """
    Calculates the difference between the C_llr before and after isotonic calibration.

    :param llrs: a numpy array of LLRs
    :param y: a numpy array of labels (0 or 1)
    :param weights: the relative weights of the classes
    :return: CLLR_cal, the difference after isotonic calibration
    """
    llrs, y = llr_data.llrs, llr_data.labels
    if y is None:
        raise ValueError('Labels are required to compute C_llr')

    cllr_min_val = cllr_min(LLRData(features=llrs, labels=y), weights)
    cllr_val = cllr(LLRData(features=llrs, labels=y), weights)
    return cllr_val - cllr_min_val
