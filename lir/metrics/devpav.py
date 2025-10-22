import numpy as np

from lir.algorithms.isotonic_regression import IsotonicCalibrator
from lir.util import Xy_to_Xn, logodds_to_odds


def _calcsurface(c1: (float, float), c2: (float, float)) -> float:
    """
    Helperfunction that calculates the desired surface for two xy-coordinates
    """
    # step 1: calculate intersection (xs, ys) of straight line through coordinates with identity line (if slope (a) = 1,
    # there is no intersection and surface of this parrellogram is equal to deltaY * deltaX)

    x1, y1 = c1
    x2, y2 = c2
    a = (y2 - y1) / (x2 - x1)

    if a == 1:
        # dan xs equals +/- Infinite en is er there is no intersection with the identity line

        # the surface of the parallellogram is:
        surface = (x2 - x1) * np.abs(y1 - x1)

    elif a < 0:
        raise ValueError(
            f"slope is negative; impossible for PAV-transform. Coordinates are {c1} and {c2}. Calculated slope is {a}"
        )
    else:
        # than xs is finite:
        b = y1 - a * x1
        xs = b / (1 - a)
        # xs

        # step 2: check if intersection is located within line segment c1 and c2.
        if x1 < xs and x2 >= xs:
            # then intersection is within
            # (situation 1 of 2) if y1 <= x1 than surface is:
            if y1 <= x1:
                surface = (
                    0.5 * (xs - y1) * (xs - x1)
                    - 0.5 * (xs - x1) * (xs - x1)
                    + 0.5 * (y2 - xs) * (x2 - xs)
                    - 0.5 * (x2 - xs) * (x2 - xs)
                )
            else:
                # (situation 2 of 2) than y1 > x1, and surface is:
                surface = (
                    0.5 * (xs - x1) ** 2
                    - 0.5 * (xs - y1) * (xs - x1)
                    + 0.5 * (x2 - xs) ** 2
                    - 0.5 * (x2 - xs) * (y2 - xs)
                )
                # dit is the same as 0.5 * (xs - x1) * (xs - y1) - 0.5 * (xs - y1) * (xs - y1)
                # + 0.5 * (y2 - xs) * (x2 - xs) - 0.5 * (y2 - xs) * (y2 - xs) + 0.5 * (y1 - x1) * (y1 - x1)
                # + 0.5 * (x2 - y2) * (x2 -y2)
        else:  # then intersection is not within line segment
            # if (situation 1 of 4) y1 <= x1 AND y2 <= x1, and surface is
            if y1 <= x1 and y2 <= x1:
                surface = 0.5 * (y2 - y1) * (x2 - x1) + (x1 - y2) * (x2 - x1) + 0.5 * (x2 - x1) * (x2 - x1)
            elif y1 > x1:  # (situation 2 of 4) then y1 > x1, and surface is
                surface = 0.5 * (x2 - x1) * (x2 - x1) + (y1 - x2) * (x2 - x1) + 0.5 * (y2 - y1) * (x2 - x1)
            elif y1 <= x1 and y2 > x1:  # (situation 3 of 4). This should be the last possibility.
                surface = 0.5 * (y2 - y1) * (x2 - x1) - 0.5 * (y2 - x1) * (y2 - x1) + 0.5 * (x2 - y2) * (x2 - y2)
            else:
                # situation 4 of 4 : this situation should never appear. There is a fourth sibution as situation 3,
                # but than above the identity line. However, this is impossible by definition of a
                # PAV-transform (y2 > x1).
                raise ValueError(f"unexpected coordinate combination: ({x1}, {y1}) and ({x2}, {y2})")
    return surface


def _devpavcalculator(lrs, pav_lrs, y):
    """
    function that calculates davPAV for a PAVresult for SSLRs and DSLRs  een PAV transformatie de devPAV uitrekent
    Input: Lrs = np.array met LR-waarden. pav_lrs = np.array met uitkomst van PAV-transformatie op lrs. y = np.array met
        labels (1 voor H1 en 0 voor H2)
    Output: devPAV value

    """
    DSLRs, SSLRs = Xy_to_Xn(lrs, y)
    DSPAVLRs, SSPAVLRs = Xy_to_Xn(pav_lrs, y)
    PAVresult = np.concatenate([SSPAVLRs, DSPAVLRs])
    Xen = np.concatenate([SSLRs, DSLRs])

    # order coordinates based on x's then y's and filtering out identical datapoints
    data = np.unique(np.array([Xen, PAVresult]), axis=1)
    Xen = data[0, :]
    Yen = data[1, :]

    # pathological cases
    # first one of four: PAV-transform has a horizonal line to log(X) = -Inf as to log(X) = Inf
    if Yen[0] != 0 and Yen[-1] != np.inf and Xen[-1] == np.inf and Xen[-1] == np.inf:
        return np.inf

    # second of four: PAV-transform has a horizontal line to log(X) = -Inf
    if Yen[0] != 0 and Xen[0] == 0 and Yen[-1] == np.inf:
        return np.inf

    # third of four: PAV-transform has a horizontal line to log(X) = Inf
    if Yen[0] == 0 and Yen[-1] != np.inf and Xen[-1] == np.inf:
        return np.inf

    # forth of four: PAV-transform has one vertical line from log(Y) = -Inf to log(Y) = Inf
    wh = (Yen == 0) | (Yen == np.inf)
    if np.sum(wh) == len(Yen):
        return np.nan

    else:
        # then it is not a  pathological case with weird X-values and devPAV can be calculated

        # filtering out -Inf or 0 Y's
        wh = (Yen > 0) & (Yen < np.inf)
        Xen = np.log10(Xen[wh])
        Yen = np.log10(Yen[wh])
        # create an empty list with size (len(Xen))
        devPAVs = [None] * len(Xen)
        # sanity check
        if len(Xen) == 0:
            return np.nan
        elif len(Xen) == 1:
            return abs(Xen - Yen)
        # than calculate devPAV
        else:
            deltaX = Xen[-1] - Xen[0]
            surface = 0
            for i in range(1, (len(Xen))):
                surface = surface + _calcsurface((Xen[i - 1], Yen[i - 1]), (Xen[i], Yen[i]))
                devPAVs[i - 1] = _calcsurface((Xen[i - 1], Yen[i - 1]), (Xen[i], Yen[i]))
            # return(list(surface/a, PAVresult, Xen, Yen, devPAVs))
            return surface / deltaX


def devpav(llrs: np.ndarray, y: np.ndarray) -> float:
    """
    calculates devPAV for LR data under H1 and H2.
    """
    if sum(y) == len(y) or sum(y) == 0:
        raise ValueError("devpav: illegal input: at least one value is required for each class")
    cal = IsotonicCalibrator()
    pavllrs = cal.fit_transform(llrs, y)
    return _devpavcalculator(logodds_to_odds(llrs), logodds_to_odds(pavllrs), y)
