import numpy as np

from lir.algorithms.isotonic_regression import IsotonicCalibrator
from lir.data.models import LLRData
from lir.util import Xy_to_Xn, logodds_to_odds


def _calcsurface(c1: tuple[float, float], c2: tuple[float, float]) -> float:
    """
    Calculate the desired surface for two xy-coordinates.

    Parameters
    ----------
    c1 : tuple[float, float]
        First coordinate as `(x, y)`.
    c2 : tuple[float, float]
        Second coordinate as `(x, y)`.

    Returns
    -------
    float
        Surface between line segment `c1`-`c2` and the identity line.
    """
    # step 1: calculate intersection (xs, ys) of straight line through coordinates with identity line (if slope (a) = 1,
    # there is no intersection and surface of this parallelogram is equal to deltaY * deltaX)

    x1, y1 = c1
    x2, y2 = c2
    a = (y2 - y1) / (x2 - x1)

    if a == 1:
        # then xs equals +/- Infinite and there is no intersection with the identity line

        # the surface of the parallelogram is:
        surface = (x2 - x1) * np.abs(y1 - x1)

    elif a < 0:
        raise ValueError(
            f'slope is negative; impossible for PAV-transform. Coordinates are {c1} and {c2}. Calculated slope is {a}'
        )
    else:
        # then xs is finite:
        b = y1 - a * x1
        xs = b / (1 - a)

        # step 2: check if intersection is located within line segment c1 and c2.
        if x1 < xs and x2 >= xs:
            # then intersection is within
            # (situation 1 of 2) if y1 <= x1 then surface is:
            if y1 <= x1:
                surface = (
                    0.5 * (xs - y1) * (xs - x1)
                    - 0.5 * (xs - x1) * (xs - x1)
                    + 0.5 * (y2 - xs) * (x2 - xs)
                    - 0.5 * (x2 - xs) * (x2 - xs)
                )
            else:
                # (situation 2 of 2) then y1 > x1, and surface is:
                surface = (
                    0.5 * (xs - x1) ** 2
                    - 0.5 * (xs - y1) * (xs - x1)
                    + 0.5 * (x2 - xs) ** 2
                    - 0.5 * (x2 - xs) * (y2 - xs)
                )
                # this is the same as 0.5 * (xs - x1) * (xs - y1) - 0.5 * (xs - y1) * (xs - y1)
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
                # situation 4 of 4 : this situation should never appear. There is a fourth situation as situation 3,
                # but then above the identity line. However, this is impossible by definition of a
                # PAV-transform (y2 > x1).
                raise ValueError(f'unexpected coordinate combination: ({x1}, {y1}) and ({x2}, {y2})')
    return surface


def _devpavcalculator(lrs: np.ndarray, pav_lrs: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate devPAV for PAV-transformed LRs.

    Parameters
    ----------
    lrs : np.ndarray
        Array of LR values.
    pav_lrs : np.ndarray
        Array of LRs after PAV transformation.
    y : np.ndarray
        Array of labels (`1` for H1 and `0` for H2).

    Returns
    -------
    float
        DevPAV value.
    """
    # split same-source and different-source LRs and PAV-LRs
    ds_lrs, ss_lrs = Xy_to_Xn(lrs, y)
    ds_pav_lrs, ss_pav_lrs = Xy_to_Xn(pav_lrs, y)
    pav_result = np.concatenate([ss_pav_lrs, ds_pav_lrs])
    all_lrs = np.concatenate([ss_lrs, ds_lrs])

    # order coordinates based on all_lrs then pav_result and filter out identical datapoints
    data = np.unique(np.array([all_lrs, pav_result]), axis=1)
    all_lrs_sorted = data[0, :]
    pav_result_sorted = data[1, :]

    # Check for pathological case where devPAV = undefined
    # PAV-transform has one vertical line from log(pav_result_sorted) = -Inf to log(pav_result_sorted) = Inf;
    #   there is no overlap in the DS and SS LRs.
    wh = (pav_result_sorted == 0) | (pav_result_sorted == np.inf)
    if np.sum(wh) == len(pav_result_sorted):
        return np.nan

    # Check for pathological cases where devPAV is infinite.
    # devPAV is infinite if we have a horizontal line either to log(all_lrs_sorted) = -inf or
    #   to log(all_lrs_sorted) = inf (or both).
    if (pav_result_sorted[0] != 0 and all_lrs_sorted[0] == 0) or (
        pav_result_sorted[-1] != np.inf and all_lrs_sorted[-1] == np.inf
    ):
        return np.inf

    # filtering out Inf or 0 Y's and taking log10
    within_range_index = (pav_result_sorted > 0) & (pav_result_sorted < np.inf)
    llrs_sorted = np.log10(all_lrs_sorted[within_range_index])
    log_pav_result_sorted = np.log10(pav_result_sorted[within_range_index])

    # Check two pathological cases
    #   1. Return nan for no data left
    if len(llrs_sorted) == 0:
        return np.nan

    #   2. Return abs(x - y) if only one datapoint left
    if len(llrs_sorted) == 1:
        return abs(llrs_sorted[0] - log_pav_result_sorted[0])

    # Actual devPAV calculation
    surface = sum(
        _calcsurface((llrs_sorted[i - 1], log_pav_result_sorted[i - 1]), (llrs_sorted[i], log_pav_result_sorted[i]))
        for i in range(1, len(llrs_sorted))
    )

    delta_llrs = llrs_sorted[-1] - llrs_sorted[0]
    return surface / delta_llrs


def devpav(llrs: LLRData) -> float:
    """
    Calculate devPAV for LR data under H1 and H2.

    Parameters
    ----------
    llrs : LLRData
        LLRs and their metadata, wrapped in an `LLRData` object.

    Returns
    -------
    float
        DevPAV score.
    """
    hypothesis = llrs.check_both_hypotheses()
    cal = IsotonicCalibrator()
    pavllrs = cal.fit_apply(llrs)
    return _devpavcalculator(logodds_to_odds(llrs.llrs), logodds_to_odds(pavllrs.llrs), hypothesis)
