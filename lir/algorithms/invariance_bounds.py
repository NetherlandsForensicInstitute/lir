"""
Extrapolation bounds on LRs using the Invariance Verification method by Alberink et al. (2025)

See:
[-] A transparent method to determine limit values for Likelihood Ratio systems, by
    Ivo Alberink, Jeannette Leegwater, Jonas Malmborg, Anders Nordgaard, Marjan Sjerps, Leen van der Ham
    In: Submitted for publication in 2025.
"""

import matplotlib.pyplot as plt
import numpy as np

from lir.bounding import LLRBounder
from lir.data.models import LLRData


def plot_invariance_delta_functions(
    llrdata: LLRData,
    llr_threshold_range: tuple[float, float] | None = None,
    step_size: float = 0.001,
    ax: plt.Axes | None = None,
) -> None:
    """
    Returns a figure of the Invariance Verification delta functions along with the upper and lower bounds of the LRs.

    :param lrs: an array of LRs
    :param y: an array of ground-truth labels (values 0 for Hd or 1 for Hp);
        must be of the same length as `lrs`
    :param llr_threshold_range: lower limit and upper limit for the LLRs to include in the figure
    :param step_size: required accuracy on a base-10 logarithmic scale
    :param ax: matplotlib axes
    """

    llrs, y = llrdata.llrs, llrdata.labels
    if y is None:
        raise ValueError('Ground-truth labels are required to plot invariance delta functions.')

    if ax is None:
        _, ax = plt.subplots()

    if llr_threshold_range is None:
        llr_threshold_range = (np.min(llrs) - 0.5, np.max(llrs) + 0.5)

    llr_threshold = np.arange(*llr_threshold_range, step_size)

    lower_bound, upper_bound, delta_low, delta_high = calculate_invariance_bounds(llrdata, llr_threshold)

    # plot the delta-functions and the 0-line
    lower_llr = np.round(lower_bound, 2)
    upper_llr = np.round(upper_bound, 2)
    ax.plot(
        llr_threshold,
        delta_low,
        '--',
        label=r'$\Delta_{lower}$ is 0 at ' + str(lower_llr),
    )
    ax.plot(
        llr_threshold,
        delta_high,
        '-',
        label=r'$\Delta_{upper}$ is 0 at ' + str(upper_llr),
    )
    ax.axhline(color='k', linestyle='dotted')
    # Some more formatting
    ax.legend(loc='upper left')
    ax.set_xlabel('log10(LR)')
    ax.set_ylabel(r'$\Delta$-value')


def calculate_invariance_bounds(
    llrdata: LLRData,
    llr_threshold: np.ndarray | None = None,
    step_size: float = 0.001,
    substitute_extremes: tuple[float, float] = (-20, 20),
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """
    Returns the upper and lower Invariance Verification bounds of the LRs.

    :param lrdata: an instance of LLRData containing LLRs and ground-truth labels
    :param llr_threshold: predefined values of LLRs as possible bounds
    :param step_size: required accuracy on a base-10 logarithmic scale
    :param substitute_extremes: (tuple of scalars) substitute for extreme LLRs, i.e.
        LLRs smaller than the lower value or greater than the upper value are clipped
    """
    llrs, y = llrdata.llrs, llrdata.labels

    # remove LLRs that are too extreme by clipping them to the substitute extremes
    sanitized_llrs = llrs.copy()
    np.clip(sanitized_llrs, substitute_extremes[0], substitute_extremes[1], out=sanitized_llrs)

    # determine the range of LLRs to be considered
    if llr_threshold is None:
        llr_threshold_range = (min(0, np.min(sanitized_llrs)), max(0, np.max(sanitized_llrs)) + step_size)
        llr_threshold = np.arange(*llr_threshold_range, step_size)

    # calculate the two delta functions
    sanitized_llrdata = LLRData(features=sanitized_llrs, labels=y)
    delta_low, delta_high = calculate_invariance_delta_functions(sanitized_llrdata, llr_threshold)

    # find the LLRs closest to LLR=0 where the functions become negative & convert them to LRs
    # if no negatives are found, use the maximum H1-LR in case of upper bound & minimum H2-LR in case of lower bound
    delta_high_negative = np.where(delta_high < 0)[0]
    if not any(delta_high_negative):
        upper_bound = np.max(sanitized_llrs[y == 1])
    else:
        upper_bound_index = delta_high_negative[0] - 1
        upper_bound = llr_threshold[upper_bound_index]

    delta_low_negative = np.where(delta_low < 0)[0]
    if not any(delta_low_negative):
        lower_bound = np.min(sanitized_llrs[y == 0])
    else:
        lower_bound_index = delta_low_negative[-1] + 1
        lower_bound = llr_threshold[lower_bound_index]

    # Check for bounds on the wrong side of 0 (or 1 in LR-space). This may occur for badly
    # performing LR systems, e.g. if the delta function is always below zero.
    lower_bound = float(min(lower_bound, 1.0))
    upper_bound = float(max(upper_bound, 1.0))

    return lower_bound, upper_bound, delta_low, delta_high


def calculate_invariance_delta_functions(llrdata: LLRData, llr_threshold: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculates the Invariance Verification delta functions for a set of LRs at given threshold values.

    :param lrs: an array of LRs
    :param y: an array of ground-truth labels (values 0 for Hd or 1 for Hp);
        must be of the same length as `lrs`
    :param llr_threshold: an array of threshold LLRs
    :returns: two arrays of delta-values, at all threshold LR values
    """
    llrs, y = llrdata.llrs, llrdata.labels
    # fix the value used for the beta distributions at 1/2 (Jeffreys prior)
    beta_parameter = 1 / 2

    # for all possible llr_threshold values, count how many of the lrs are larger or equal to them for both h1 and h2
    llrs_h1 = llrs[y == 1]
    llrs_h2 = llrs[y == 0]
    llr_h1_2d = np.tile(np.expand_dims(llrs_h1, 1), (1, llr_threshold.shape[0]))
    llr_h2_2d = np.tile(np.expand_dims(llrs_h2, 1), (1, llr_threshold.shape[0]))
    success_h1 = np.sum(llr_h1_2d >= llr_threshold, axis=0)
    success_h2 = np.sum(llr_h2_2d >= llr_threshold, axis=0)

    # use the as inputs for calculations of the probabilities
    prob_h1_above_grid = (success_h1 + beta_parameter) / (len(llrs_h1) + 2 * beta_parameter)
    prob_h2_above_grid = (success_h2 + beta_parameter) / (len(llrs_h2) + 2 * beta_parameter)
    prob_h1_below_grid = 1 - prob_h1_above_grid
    prob_h2_below_grid = 1 - prob_h2_above_grid

    # calculate the delta-functions for all the llr_threshold values
    delta_high = np.log10(prob_h1_above_grid) - np.log10(prob_h2_above_grid) - llr_threshold
    delta_low = llr_threshold - np.log10(prob_h1_below_grid) + np.log10(prob_h2_below_grid)

    return delta_low, delta_high


class IVBounder(LLRBounder):
    """
    Class that, given an LR system, outputs the same LRs as the system but bounded by the Invariance Verification
    bounds as described in:
    A transparent method to determine limit values for Likelihood Ratio systems, by
    Ivo Alberink, Jeannette Leegwater, Jonas Malmborg, Anders Nordgaard, Marjan Sjerps, Leen van der Ham
    In: Submitted for publication in 2025.
    """

    def calculate_bounds(self, llrdata: LLRData) -> tuple[float | None, float | None]:
        lower_llr_bound, upper_llr_bound = calculate_invariance_bounds(llrdata)[:2]
        return lower_llr_bound, upper_llr_bound
