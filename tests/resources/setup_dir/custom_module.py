import numpy as np


def double_logodds(p):
    """
    Custom to-LLR module, used to test that modules in the setup directory are importable.

    Doubles the log-odds, mimicking the interface of `lir.util.probability_to_logodds`.
    """
    with np.errstate(divide='ignore', invalid='raise'):
        return 2.0 * (np.log10(p) - np.log10(1.0 - p))
