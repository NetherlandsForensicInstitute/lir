"""
Normalized Bayes Error-rate (NBE)

See:
[-] Peter Vergeer, Andrew van Es, Arent de Jongh, Ivo Alberink and Reinoud
    Stoel, Numerical likelihood ratios outputted by LR systems are often based
    on extrapolation: When to stop extrapolating? In: Science and Justice 56
    (2016) 482–491.
"""

import matplotlib.pyplot as plt
import numpy as np

from lir.bounding import LLRBounder
from lir.data.models import LLRData
from lir.util import logodds_to_odds


def plot_nbe(
    llrdata: LLRData,
    log_lr_threshold_range: tuple[float, float] | None = None,
    add_misleading: int = 1,
    step_size: float = 0.01,
    ax: plt.Axes = plt,  # type: ignore
) -> None:
    llrs = llrdata.llrs
    y = llrdata.labels
    if y is None:
        raise ValueError('LLRData must contain labels to plot NBE.')

    if log_lr_threshold_range is None:
        log_lr_threshold_range = (np.min(llrs) - 0.5, np.max(llrs) + 0.5)

    log_lr_threshold = np.arange(*log_lr_threshold_range, step_size)
    lr_threshold = np.power(10, log_lr_threshold)

    eu_neutral = calculate_expected_utility(np.ones_like(llrs), y, lr_threshold)
    eu_system = calculate_expected_utility(logodds_to_odds(llrs), y, lr_threshold, add_misleading)

    ax.plot(log_lr_threshold, np.log10(eu_neutral / eu_system))

    ax.set_xlabel('log$_{10}$(threshold LR)')
    ax.set_ylabel('log$_{10}$(expected utility ratio)')
    ax.set_xlim(log_lr_threshold_range)
    ax.grid(True, linestyle=':')


def elub(
    llrs: np.ndarray,
    y: np.ndarray,
    add_misleading: int = 1,
    step_size: float = 0.01,
    substitute_extremes: tuple[float, float] = (-9, 9),
) -> tuple[float, float]:
    """
    Returns the empirical upper and lower bound log10-LRs (ELUB LLRs).

    :param llrs: an array of log10-LRs
    :param y: an array of ground-truth labels (values 0 for Hd or 1 for Hp);
        must be of the same length as `llrs`
    :param add_misleading: the number of consequential misleading LLRs to be added
        to both sides (labels 0 and 1)
    :param step_size: required accuracy on a 10-base logarithmic scale
    :param substitute_extremes: tuple of scalars: substitute for extreme LRs, i.e.
        LRs of 0 and inf are substituted by these values
    """

    # remove LLRs of -infinity and +infinity
    sanitized_llrs = llrs
    sanitized_llrs[sanitized_llrs < substitute_extremes[0]] = substitute_extremes[0]
    sanitized_llrs[sanitized_llrs > substitute_extremes[1]] = substitute_extremes[1]

    # determine the range of LLRs to be considered, using dataset sizes: LB > -log10(size(Hp)+1), UB < log10(size(Hd)+1)
    llr_min = max(np.min(sanitized_llrs), -np.log10(np.sum(y) + 1))
    llr_max = min(np.max(sanitized_llrs), np.log10(np.sum(1 - y) + 1))
    llr_steps_min = min(0, int(np.floor_divide(llr_min, step_size)))
    llr_steps_max = max(0, int((np.floor_divide(llr_max, step_size)) + 1))
    llr_threshold = np.linspace(llr_steps_min * step_size, llr_steps_max * step_size, llr_steps_max - llr_steps_min + 1)

    # calculate the ratio of the expected utilities
    eu_neutral = calculate_expected_utility(np.ones(len(sanitized_llrs)), y, 10**llr_threshold)
    eu_system = calculate_expected_utility(10**sanitized_llrs, y, 10**llr_threshold, add_misleading)
    eu_ratio = eu_neutral / eu_system

    # find threshold LLRs which have utility ratio < 1 (only utility ratio >= 1 is acceptable)
    eu_negative_left = llr_threshold[(llr_threshold <= 0) & (eu_ratio < 1)]
    eu_negative_right = llr_threshold[(llr_threshold >= 0) & (eu_ratio < 1)]

    # use the most conservative LLR as bound (closest to 0, assuming all are on the expected size of 0)
    lower_bound = np.max(eu_negative_left + step_size, initial=np.min(llr_threshold))
    upper_bound = np.min(eu_negative_right - step_size, initial=np.max(llr_threshold))

    # Check for bounds on the wrong side of 0. This may occur for badly
    # performing LR systems, e.g. if expected utility is always below neutral.
    lower_bound = min(lower_bound, 0)
    upper_bound = max(upper_bound, 0)

    return lower_bound, upper_bound


def calculate_expected_utility(
    lrs: np.ndarray, y: np.ndarray, threshold_lrs: np.ndarray, add_misleading: int = 0
) -> float:
    """
    Calculates the expected utility of a set of LRs for a given threshold.

    :param lrs: an array of LRs
    :param y: an array of ground-truth labels (values 0 for Hd or 1 for Hp);
        must be of the same length as `lrs`
    :param threshold_lrs: an array of threshold lrs: minimum LR for acceptance
    :returns: an array of utility values, one element for each threshold LR
    """
    m_accept = lrs.reshape(len(lrs), 1) > threshold_lrs.reshape(1, len(threshold_lrs))

    if add_misleading > 0:
        n_elems = len(threshold_lrs) * add_misleading
        m_accept = np.concatenate(
            [
                m_accept,
                np.zeros(n_elems).reshape(add_misleading, len(threshold_lrs)),
                np.ones(n_elems).reshape(add_misleading, len(threshold_lrs)),
            ]
        )
        y = np.concatenate([y, np.ones(add_misleading), np.zeros(add_misleading)])

    eu = 1 - np.average(m_accept[y == 1], axis=0) + threshold_lrs * np.average(m_accept[y == 0], axis=0)
    return eu


class ELUBBounder(LLRBounder):
    """
    Class that, given an LR system, outputs the same LRs as the system but bounded by the Empirical Upper and Lower
    Bounds as described in
    P. Vergeer, A. van Es, A. de Jongh, I. Alberink, R.D. Stoel,
    Numerical likelihood ratios outputted by LR systems are often based on extrapolation:
    when to stop extrapolating?
    Sci. Justics 56 (2016) 482-491

    # MATLAB code from the authors:

    # clear all; close all;
    # llrs_hp=csvread('...');
    # llrs_hd=csvread('...');
    # start=-7; finish=7;
    # rho=start:0.01:finish; theta=10.^rho;
    # nbe=[];
    # for k=1:length(rho)
    #     if rho(k)<0
    #         llrs_hp=[llrs_hp;rho(k)];
    #         nbe=[nbe;(theta(k)^(-1))*mean(llrs_hp<=rho(k))+...
    #             mean(llrs_hd>rho(k))];
    #     else
    #         llrs_hd=[llrs_hd;rho(k)];
    #         nbe=[nbe;theta(k)*mean(llrs_hd>=rho(k))+...
    #             mean(llrs_hp<rho(k))];
    #     end
    # end
    # plot(rho,-log10(nbe)); hold on;
    # plot([start finish],[0 0]);
    # a=rho(-log10(nbe)>0);
    # empirical_bounds=[min(a) max(a)]
    """

    def calculate_bounds(self, llrs: np.ndarray, labels: np.ndarray) -> tuple[float | None, float | None]:
        return elub(llrs, labels, add_misleading=1)
