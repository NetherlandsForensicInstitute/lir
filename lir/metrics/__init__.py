from typing import Tuple

import numpy as np

from lir.algorithms.isotonic_regression import IsotonicCalibrator
from lir.util import Xy_to_Xn, odds_to_probability, logodds_to_odds


def cllr(
    lrs: np.ndarray, y: np.ndarray, weights: Tuple[float, float] = (1, 1)
) -> float:
    """
    Calculates a log likelihood ratio cost (C_llr) for a series of likelihood
    ratios.

    Nico Brümmer and Johan du Preez, Application-independent evaluation of speaker detection, In: Computer Speech and
    Language 20(2-3), 2006.

    :param lrs: a numpy array of LRs
    :param y: a numpy array of labels (0 or 1)
    :param weights: the relative weights of the classes
    :return: CLLR, the log likelihood ratio cost
    """

    # ignore errors:
    #   divide -> ignore divide by zero
    #   over -> ignore scalar overflow
    with np.errstate(divide="ignore", over="ignore"):
        lrs0, lrs1 = Xy_to_Xn(lrs, y)
        cllr0 = weights[0] * np.mean(np.log2(1 + lrs0)) if weights[0] > 0 else 0
        cllr1 = weights[1] * np.mean(np.log2(1 + 1 / lrs1)) if weights[1] > 0 else 0
        return (cllr0 + cllr1) / sum(weights)


def cllr_min(
    lrs: np.ndarray, y: np.ndarray, weights: Tuple[float, float] = (1, 1)
) -> float:
    """
    Estimates the discriminative power from a collection of likelihood ratios.

    :param lrs: a numpy array of LRs
    :param y: a numpy array of labels (0 or 1)
    :param weights: the relative weights of the classes
    :return: CLLR_min, a measure of discrimination
    """
    cal = IsotonicCalibrator()
    llrmin = cal.fit_transform(odds_to_probability(lrs), y)
    return cllr(logodds_to_odds(llrmin), y, weights)
