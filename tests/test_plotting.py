#!/usr/bin/env python3
import warnings

import numpy as np
import unittest

import matplotlib.pyplot as plt
import pytest

from lir import plotting
from lir.algorithms.logistic_regression import LogitCalibrator
from lir.util import odds_to_probability, odds_to_logodds


class TestPlotting(unittest.TestCase):
    def test_contexts(self):
        #warnings.simplefilter("error")
        lrs = np.array([.5, .5, .5, 1, 1, 2, 2, 2])
        llrs = odds_to_logodds(lrs)
        y = np.array([0, 0, 1, 0, 1, 0, 1, 1])

        # inside context
        with plotting.axes() as ax:
            ax.pav(llrs, y)
            ax.title("PAV plot using savefig()")

        # without context
        fig = plt.figure()
        plotting.pav(llrs, y)
        plt.title("simple call with full control")
        plt.close(fig)
        
        # sub plots
        fig, axs = plt.subplots(2)
        plotting.pav(llrs, y, ax=axs[0])
        plotting.ece(llrs, y, ax=axs[1])
        plt.close(fig)

    def test_calls(self):
        lrs = np.array([.5, .5, .5, 1, 1, 2, 2, 2, np.inf, 0])
        llrs = odds_to_logodds(lrs)
        scores = odds_to_probability(lrs)
        y = np.array([0, 0, 1, 0, 1, 0, 1, 1, 1, 0])
        finite_index = (lrs > 0) & (lrs < np.inf)

        with plotting.axes() as ax:
            ax.pav(llrs, y)

        with plotting.axes() as ax:
            ax.ece(llrs, y)

        with plotting.axes() as ax:
            ax.tippett(llrs, y)

        with pytest.raises(Exception):
            with plotting.axes() as ax:
                ax.nbe(llrs, y)

        with plotting.axes() as ax:
            ax.nbe(llrs[finite_index], y[finite_index])

        with plotting.axes() as ax:
            ax.lr_histogram(llrs[finite_index], y[finite_index])

        with plotting.axes() as ax:
            ax.score_distribution(scores, y)

        cal = LogitCalibrator()
        cal.fit(scores, y)
        with plotting.axes() as ax:
            ax.calibrator_fit(cal)


if __name__ == '__main__':
    unittest.main()
