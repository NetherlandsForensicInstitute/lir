from typing import Any

import numpy as np

from lir.algorithms.llr_overestimation import calc_llr_overestimation


def llr_overestimation(llrs: np.ndarray, y: np.ndarray, **kwargs: Any) -> float:
    """
    Calculate the mean absolute value of the LLR-overestimation.

    Parameters
    ----------
    llrs : np.ndarray
        Array of log-likelihood ratios.
    y : np.ndarray
        Array of labels (`1` for H1 and `0` for H2).
    **kwargs : Any
        Additional keyword arguments forwarded to `calc_llr_overestimation`.

    Returns
    -------
    float
        Mean absolute value of the overestimation grid, or `np.nan` when unavailable.
    """
    _, llr_overestimation_grid, _ = calc_llr_overestimation(llrs, y, num_fids=0, **kwargs)
    if llr_overestimation_grid is not None and llr_overestimation_grid.all():
        return float(np.mean(np.abs(llr_overestimation_grid)))
    else:
        return np.nan
