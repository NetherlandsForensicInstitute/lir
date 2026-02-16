"""
Empirical cross-entropy (ECE).

The discrimination and calibration of the LRs reported by some systems can also
be measured separately. The empirical cross entropy (ECE) plot is a graphical
way of doing this.

ECE is computed as the average of:

- ``-P(Hp) * log2(P(Hp | LR_i))`` over all ``LR_i`` for which ``Hp`` is true, and
- ``-P(Hd) * log2(P(Hd | LR_i))`` over all ``LR_i`` for which ``Hd`` is true.

References
----------
Ramos, D. *Forensic Evidence Evaluation Using Automatic Speaker Recognition
Systems*. Ph.D. thesis, Universidad Autónoma de Madrid.

Robertson, B., Vignaux, G. A., & Berger, C. (2016). *Interpreting Evidence:
Evaluating Forensic Science in the Courtroom* (2nd ed.), pp. 96–97.
"""

from typing import Any

import numpy as np

from lir.algorithms.isotonic_regression import IsotonicCalibrator
from lir.data.models import LLRData
from lir.util import (
    logodds_to_odds,
    odds_to_probability,
    probability_to_odds,
)


def plot_ece(
    ax: Any,
    llrdata: LLRData,
    log_prior_odds_range: tuple[float, float] = (-3, 3),
    show_pav: bool = True,
    ylim: str = 'neutral',
) -> None:
    """
    Generate an ECE plot for a set of LRs and corresponding ground-truth labels.

    The x-axis shows the log prior odds of a sample being drawn from class 1. The
    y-axis shows the expected cost (cross-entropy) for:

    1. A non-informative system (dotted line),
    2. The provided LR values (solid line), and
    3. The LR values after PAV transformation (Pool Adjacent Violators; dashed line).

    Parameters
    ----------
    llrdata : LLRData
        LLR data containing LLR values and corresponding labels.
    log_prior_odds_range : tuple[float, float], optional
        Range of log prior odds shown on the x-axis, given as ``(min, max)``.
    ax : matplotlib.axes.Axes, optional
        Matplotlib axes to plot into. If ``None``, the current axes are used.
    show_pav : bool, optional
        Whether to include the PAV-transformed LRs in the plot.
    ylim : {"neutral", "zoomed"}, optional
        Y-axis scaling mode:

        - ``"neutral"``: Lower limit is 0; upper limit is set slightly above the
          maximum of the non-informative reference.
        - ``"zoomed"``: Lower limit is 0; upper limit is set to approximately 10%
          above the maximum ECE value of the LRs (this may clip part of the
          non-informative reference line).
    """
    labels = llrdata.require_labels

    log_prior_odds = np.arange(*log_prior_odds_range, 0.01)
    prior_odds = np.power(10, log_prior_odds)

    # plot reference
    if ax.get_legend() is None or 'reference' not in [text.get_text() for text in ax.get_legend().get_texts()]:
        ax.plot(
            log_prior_odds,
            calculate_ece(np.ones_like(labels), labels, odds_to_probability(prior_odds)),
            linestyle=':',
            label='reference',
        )

    # plot LRs
    ece_values = calculate_ece(logodds_to_odds(llrdata.llrs), labels, odds_to_probability(prior_odds))
    ax.plot(
        log_prior_odds,
        ece_values,
        linestyle='-',
        label='LRs',
    )

    if show_pav:
        # plot PAV LRs
        pav_llrs = IsotonicCalibrator().fit_apply(llrdata)
        ax.plot(
            log_prior_odds,
            calculate_ece(logodds_to_odds(pav_llrs.llrs), labels, odds_to_probability(prior_odds)),
            linestyle='--',
            label='PAV LRs',
        )

    ax.set_xlabel('prior log$_{10}$(odds)')
    ax.set_ylabel('empirical cross-entropy')

    if ylim == 'neutral':
        ax.set_ylim((0, None))
    elif ylim == 'zoomed':
        ylim_value = max(ece_values) * 1.1
        ax.set_ylim((0, ylim_value))
    else:
        raise ValueError(f"ylim must be one of ['neutral', 'zoomed'], but got `{ylim}`")

    ax.set_xlim(log_prior_odds_range)
    ax.legend()
    ax.grid(True, linestyle=':')


def calculate_ece(lrs: np.ndarray, y: np.ndarray, priors: np.ndarray) -> np.ndarray:
    """Calculate empirical cross-entropy (ECE) of a set of LRs and corresponding ground-truth labels.

    An entropy is calculated for each element of `priors`.

    :param lrs: an array of LRs
    :param y: an array of ground-truth labels of the LRs (values 0 for Hd or 1
        for Hp); must be of the same length as `lrs`.
    :param priors: an array of prior probabilities of the samples being drawn
        from class 1 (values in range [0..1])
    :returns: an array of entropy values of the same length as `priors`
    """
    assert np.all(lrs >= 0), 'invalid input for LR values'
    assert np.all(np.unique(y) == np.array([0, 1])), 'label set must be [0, 1]'

    prior_odds = np.repeat(probability_to_odds(priors), len(lrs)).reshape((len(priors), len(lrs)))
    posterior_odds = prior_odds * lrs
    posterior_p = odds_to_probability(posterior_odds)

    with np.errstate(divide='ignore'):
        ece0 = -(1 - priors.reshape((len(priors), 1))) * np.log2(1 - posterior_p[:, y == 0])
        ece1 = -priors.reshape((len(priors), 1)) * np.log2(posterior_p[:, y == 1])

    ece0[np.isnan(ece0)] = np.inf
    ece1[np.isnan(ece1)] = np.inf

    avg0 = np.average(ece0, axis=1)
    avg1 = np.average(ece1, axis=1)

    return avg0 + avg1
