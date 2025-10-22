"""
Empirical Cross Entrpy (ECE)

The discrimination and calibration of the LRs reported by some systems can also
be measured separately. The empirical cross entropy (ECE) plot is a graphical
way of doing this.

The ECE is the average of -P(Hp) * log2(P(Hp|LRi)) for all LRi when Hp is true,
and -P(Hd) * log2(P(Hd|LRi)) for all LRi when Hd is true.

See:
[-] D. Ramos, Forensic evidence evaluation using automatic speaker recognition
    systems. Ph.D. Thesis. Universidad Autonoma de Madrid.
[-] Bernard Robertson, G.A. Vignaux and Charles Berger, Interpreting Evidence:
    Evaluating Forensic Science in the Courtroom, 2nd edition, 2016, pp. 96-97.
"""

import matplotlib.pyplot as plt
import numpy as np

from lir.algorithms.isotonic_regression import IsotonicCalibrator
from lir.util import (
    odds_to_probability,
    logodds_to_odds,
    probability_to_odds,
)


def plot_ece(llrs, labels, log_prior_odds_range=None, ax=plt):
    """
    Generates an ECE plot for a set of LRs and corresponding ground-truth
    labels.

    The x-axis indicates the log prior odds of a sample being drawn from class
    1; the y-axis shows the entropy for (1) a non-informative system (dotted
    line), (2) the set of LR values (line), and (3) the set of LR values after
    PAV-transformation (Pool Adjacent Violators, dashed line).

    :param lrs: an array of LRs
    :param labels: an array of ground-truth labels (values 0 for Hd or 1 for Hp);
        must be of the same length as `lrs`
    :param log_prior_odds_range: the range of prior odds (tuple of two values,
        indicating both ends of the range on the x-axis)
    """
    if log_prior_odds_range is None:
        log_prior_odds_range = (-3, 3)

    log_prior_odds = np.arange(*log_prior_odds_range, 0.01)
    prior_odds = np.power(10, log_prior_odds)

    # plot reference
    ax.plot(
        log_prior_odds,
        calculate_ece(np.ones(len(labels)), labels, odds_to_probability(prior_odds)),
        linestyle=":",
        label="reference",
    )

    # plot LRs
    ax.plot(
        log_prior_odds,
        calculate_ece(logodds_to_odds(llrs), labels, odds_to_probability(prior_odds)),
        linestyle="-",
        label="LRs",
    )

    # plot PAV LRs
    pav_llrs = IsotonicCalibrator().fit_transform(llrs, labels)
    ax.plot(
        log_prior_odds,
        calculate_ece(
            logodds_to_odds(pav_llrs), labels, odds_to_probability(prior_odds)
        ),
        linestyle="--",
        label="PAV LRs",
    )

    ax.set_xlabel("prior log$_{10}$(odds)")
    ax.set_ylabel("empirical cross-entropy")
    ax.set_ylim((0, None))
    ax.set_xlim(log_prior_odds_range)
    ax.legend()
    ax.grid(True, linestyle=":")


def calculate_ece(lrs, y, priors):
    """
    Calculates the empirical cross-entropy (ECE) of a set of LRs and
    corresponding ground-truth labels.

    An entropy is calculated for each element of `priors`.

    :param lrs: an array of LRs
    :param y: an array of ground-truth labels of the LRs (values 0 for Hd or 1
        for Hp); must be of the same length as `lrs`.
    :param priors: an array of prior probabilities of the samples being drawn
        from class 1 (values in range [0..1])
    :returns: an array of entropy values of the same length as `priors`
    """
    assert np.all(lrs >= 0), "invalid input for LR values"
    assert np.all(np.unique(y) == np.array([0, 1])), "label set must be [0, 1]"

    prior_odds = np.repeat(probability_to_odds(priors), len(lrs)).reshape(
        (len(priors), len(lrs))
    )
    posterior_odds = prior_odds * lrs
    posterior_p = odds_to_probability(posterior_odds)

    with np.errstate(divide="ignore"):
        ece0 = -(1 - priors.reshape((len(priors), 1))) * np.log2(
            1 - posterior_p[:, y == 0]
        )
        ece1 = -priors.reshape((len(priors), 1)) * np.log2(posterior_p[:, y == 1])

    ece0[np.isnan(ece0)] = np.inf
    ece1[np.isnan(ece1)] = np.inf

    avg0 = np.average(ece0, axis=1)
    avg1 = np.average(ece1, axis=1)

    return avg0 + avg1
