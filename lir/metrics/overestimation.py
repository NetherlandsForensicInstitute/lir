from typing import Any

import numpy as np

from lir.algorithms.llr_overestimation import calc_llr_overestimation


def llr_overestimation(llrs: np.ndarray, y: np.ndarray, **kwargs: Any) -> float:
    """Calculates the mean absolute value of the LLR-overestimation."""
    _, llr_overestimation_grid, _ = calc_llr_overestimation(llrs, y, num_fids=0, **kwargs)
    if llr_overestimation_grid is not None and llr_overestimation_grid.all():
        return float(np.mean(np.abs(llr_overestimation_grid)))
    else:
        return np.nan
