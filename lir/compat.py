import numpy as np


def llr_to_lr(llrs: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        return 10**llrs
