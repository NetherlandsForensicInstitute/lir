import numpy as np

from lir.algorithms.isotonic_regression import IsotonicCalibrator
from lir.data.models import LLRData
from lir.util import Xy_to_Xn, logodds_to_odds


def cllr(llr_data: LLRData, weights: tuple[float, float] = (1, 1)) -> float:
    """Calculate a log likelihood ratio cost (C_llr) for a series of log likelihood ratios.

    Nico Brümmer and Johan du Preez, Application-independent evaluation of speaker detection, In: Computer Speech and
    Language 20(2-3), 2006.

    :param llr_data: LLRs and their metadata, wrapped in an LLRData object
    :param weights: the relative weights of the classes
    :return: CLLR, the log likelihood ratio cost
    """
    llrs, y = llr_data.llrs, llr_data.require_labels

    lrs = logodds_to_odds(llrs)

    # ignore errors:
    #   divide -> ignore divide by zero
    #   over -> ignore scalar overflow
    with np.errstate(divide='ignore', over='ignore'):
        lrs0, lrs1 = Xy_to_Xn(lrs, y)
        cllr0 = weights[0] * np.mean(np.log2(1 + lrs0)) if weights[0] > 0 else 0
        cllr1 = weights[1] * np.mean(np.log2(1 + 1 / lrs1)) if weights[1] > 0 else 0
        return float((cllr0 + cllr1) / sum(weights))


def cllr_min(llr_data: LLRData, weights: tuple[float, float] = (1, 1)) -> float:
    """Estimate the discriminative power from a collection of log likelihood ratios.

    :param llr_data: LLRs and their metadata, wrapped in an LLRData object
    :param weights: the relative weights of the classes
    :return: CLLR_min, a measure of discrimination
    """
    llrs, y = llr_data.llrs, llr_data.require_labels

    cal = IsotonicCalibrator()
    llrmin = cal.fit_transform(llrs, y)

    return cllr(LLRData(features=llrmin, labels=y), weights)


def cllr_cal(llr_data: LLRData, weights: tuple[float, float] = (1, 1)) -> float:
    """Calculate the difference between the C_llr before and after isotonic calibration.

    :param llr_data: LLRs and their metadata, wrapped in an LLRData object
    :param weights: the relative weights of the classes
    :return: CLLR_cal, the difference after isotonic calibration
    """
    cllr_min_val = cllr_min(llr_data, weights)
    cllr_val = cllr(llr_data, weights)

    return cllr_val - cllr_min_val


def llr_upper_bound(llrs: LLRData) -> float | None:
    """Provide corresponding upper bound for provided LLR data.

    When an LLRData object contains an upper bound, return it. If not, return None.

    :param llrs: LLRs and their metadata, wrapped in an LLRData object
    :return: the LLR upper bound, or None
    """
    return llrs.llr_upper_bound


def llr_lower_bound(llrs: LLRData) -> float | None:
    """Provide corresponding lower bound for provided LLR data.

    When an LLRData object contains a lower bound, return it. If not, return None.

    :param llrs: LLRs and their metadata, wrapped in an LLRData object
    :return: the LLR lower bound, or None
    """
    return llrs.llr_lower_bound
