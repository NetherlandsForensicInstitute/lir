from typing import Any

import numpy as np

from lir.algorithms.llr_overestimation import calc_llr_overestimation
from lir.util import logodds_to_odds


def llr_overestimation(llrs: np.ndarray, y: np.ndarray, **kwargs: Any) -> float:
    """
    Calculates the mean absolute value of the LLR-overestimation.
    """
    _, llr_overestimation_grid, _ = calc_llr_overestimation(
        llrs, y, num_fids=0, **kwargs
    )
    return (
        np.mean(np.abs(llr_overestimation_grid)) if llr_overestimation_grid.all() else np.nan
    )
