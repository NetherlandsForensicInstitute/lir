#!/usr/bin/env python3
import unittest

import matplotlib.pyplot as plt
import numpy as np
import pytest

from lir import plotting
from lir.aggregation import (
    PlotCalibratorFit,
    PlotECE,
    PlotLLRInterval,
    PlotLRHistogram,
    PlotNBE,
    PlotPAV,
    PlotScoreDistribution,
    PlotTippett,
)
from lir.algorithms.logistic_regression import LogitCalibrator
from lir.data.models import LLRData
from lir.util import odds_to_logodds, odds_to_probability


class TestPlotting(unittest.TestCase):
    def test_contexts(self):
        lrs = np.array([0.5, 0.5, 0.5, 1, 1, 2, 2, 2])
        llrs = odds_to_logodds(lrs)
        y = np.array([0, 0, 1, 0, 1, 0, 1, 1])
        llr_data = LLRData(features=llrs.reshape(-1, 1), labels=y)

        # without context
        fig = plt.figure()
        plotting.pav(llr_data)
        plt.title('simple call with full control')
        plt.close(fig)

        # sub plots
        fig, axs = plt.subplots(2)
        plotting.pav(llr_data, ax=axs[0])
        plotting.ece(llr_data, ax=axs[1])
        plt.close(fig)

    def test_plot_calls(self):
        """Test all plotting calls with basic data.

        This test does not check the correctness of the plots, only that they
        can be created without errors.
        """
        lrs = np.array([0.5, 0.5, 0.5, 1, 1, 2, 2, 2, np.inf, 0])
        llrs = odds_to_logodds(lrs)
        scores = odds_to_probability(lrs)
        y = np.array([0, 0, 1, 0, 1, 0, 1, 1, 1, 0])
        y_nd = y.reshape(-1, 1)
        finite_index = (lrs > 0) & (lrs < np.inf)

        # The reshape(-1, 1) is to simulate single-feature data with one sample per row
        llr_data = LLRData(features=llrs.reshape(-1, 1), labels=y)
        llr_data_finite = LLRData(features=llrs[finite_index].reshape(-1, 1), labels=y[finite_index])

        # The LLRData interval is just a 3-column ndarray where each row is:
        # [llr_point_estimate. llr_lower_bound, llr_upper_bound]
        llrs_and_interval_ndarray = np.array(
            [
                [-2.0, -2.5, -1.5],
                [0.0, -0.5, 0.5],
                [2.0, 1.5, 2.5],
                [4.0, 3.5, 4.5],
                [10.0, 9.5, 10.5],
            ]
        )
        llr_data_intervals = LLRData(features=llrs_and_interval_ndarray)

        cal = LogitCalibrator()
        cal.fit(scores, y)

        # Test that plots with invalid data raise exceptions.
        with pytest.raises(Exception):
            # This should fail because of infinite LLRs.
            PlotNBE().plot(llr_data)

            # This should fail as llr_data has no intervals.
            PlotLLRInterval().plot(llr_data)

        PlotPAV().plot(llr_data)
        PlotECE().plot(llr_data)
        PlotTippett().plot(llr_data)
        PlotNBE().plot(llr_data_finite)
        PlotLRHistogram().plot(llr_data_finite)
        PlotScoreDistribution().plot(llr_data.features, y_nd)
        PlotLLRInterval().plot(llr_data_intervals)
        PlotCalibratorFit().plot(cal)


if __name__ == '__main__':
    unittest.main()
