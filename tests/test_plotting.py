#!/usr/bin/env python3
import warnings

import numpy as np
import unittest

import matplotlib.pyplot as plt
import pytest

from lir import plotting
from lir.algorithms.logistic_regression import LogitCalibrator
from lir.data.models import LLRData
from lir.util import odds_to_probability, odds_to_logodds


class TestPlotting(unittest.TestCase):
    def test_contexts(self):
        lrs = np.array([.5, .5, .5, 1, 1, 2, 2, 2])
        llrs = odds_to_logodds(lrs)
        llr_data = LLRData(features=llrs.reshape(-1, 1))

        y = np.array([0, 0, 1, 0, 1, 0, 1, 1])

        # inside context
        with plotting.axes() as ax:
            ax.pav(llr_data, y)
            ax.title("PAV plot using savefig()")

        # without context
        fig = plt.figure()
        plotting.pav(llr_data, y)
        plt.title("simple call with full control")
        plt.close(fig)
        
        # sub plots
        fig, axs = plt.subplots(2)
        plotting.pav(llr_data, y, ax=axs[0])
        plotting.ece(llr_data, y, ax=axs[1])
        plt.close(fig)

    def test_calls(self):
        lrs = np.array([.5, .5, .5, 1, 1, 2, 2, 2, np.inf, 0])
        llrs = odds_to_logodds(lrs)
        scores = odds_to_probability(lrs)
        y = np.array([0, 0, 1, 0, 1, 0, 1, 1, 1, 0])
        finite_index = (lrs > 0) & (lrs < np.inf)

        llr_data = LLRData(features=llrs.reshape(-1, 1), labels=y)
        llr_data_finite = LLRData(features=llrs[finite_index].reshape(-1, 1))
        llrs_and_interval_ndarray = np.array([
            [-2.0, -2.5, -1.5],
            [0.0, -0.5, 0.5],
            [2.0, 1.5, 2.5],
            [4.0, 3.5, 4.5],
            [10.0, 9.5, 10.5],
        ])
        llr_data_intervals = LLRData(features=llrs_and_interval_ndarray)

        with plotting.axes() as ax:
            ax.pav(llr_data, y)

        with plotting.axes() as ax:
            ax.ece(llr_data, y)

        with plotting.axes() as ax:
            ax.tippett(llr_data, y)

        with pytest.raises(Exception):
            with plotting.axes() as ax:
                ax.nbe(llr_data, y)

        with plotting.axes() as ax:
            ax.nbe(llrs[finite_index], y[finite_index])

        with plotting.axes() as ax:
            ax.lr_histogram(llr_data_finite, y[finite_index])

        with plotting.axes() as ax:
            ax.score_distribution(scores, y)

        with plotting.axes() as ax:
            ax.llr_interval(llr_data_intervals)

        cal = LogitCalibrator()
        cal.fit(scores, y)
        with plotting.axes() as ax:
            ax.calibrator_fit(cal)


if __name__ == '__main__':
    unittest.main()
