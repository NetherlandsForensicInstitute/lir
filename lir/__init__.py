"""
LiR - Toolkit for developing, optimising and evaluating Likelihood Ratio (LR) systems.

This allows benchmarking of LR systems on different datasets, investigating impact of different
sampling schemes or techniques, and doing case-based validation and computation of case LRs.
"""

import sys

from lir.transform import Transformer as Transformer  # as required by linting, later to be replaced by __all__


def is_interactive() -> bool:
    """Determine if the LiR tool is running from the CLI and should be interactive.

    This method is used, for example, to determine if a progress bar should be shown.
    """
    return sys.stdout.isatty()
