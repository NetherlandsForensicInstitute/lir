#!/usr/bin/env python3

import numpy as np
import unittest

from lir import metrics
from lir.data.datasets.alcohol_breath_analyser import AlcoholBreathAnalyser
from lir.data.models import LLRData
from lir.plotting.expected_calibration_error import calculate_ece
from lir.util import logodds_to_odds


class TestECE(unittest.TestCase):
    def _compare_ece_cllr(self, llrs, y):
        error = calculate_ece(logodds_to_odds(llrs), y, np.array([.5, .5]))
        llr_data = LLRData(features=llrs, labels=y)
        cllr = metrics.cllr(llr_data)
        np.testing.assert_almost_equal(error[0], cllr)


    def test_cllr(self):
        breath = AlcoholBreathAnalyser(ill_calibrated=True).get_instances()
        data = [
            (breath.llrs, breath.labels),
            (np.array([-1, 1, np.inf]), np.array([0, 1, 1])),
            (np.array([-1, 1, np.inf]), np.array([0, 1, 0])),
            (np.array([-1, 1, -np.inf]), np.array([0, 1, 1])),
            (np.array([-1, 1, -np.inf]), np.array([0, 1, 0])),
        ]

        for llrs, y in data:
            self._compare_ece_cllr(llrs, y)


    def test_invalid(self):
        data = [
            (np.array([.1, 10, np.nan]), np.array([0, 1, 1]), "invalid input for LR values"),
            (np.array([.1, 10, -1]), np.array([0, 1, 0]), "invalid input for LR values"),
        ]

        for lrs, y, msg in data:
            with self.assertRaises(AssertionError) as context:
                calculate_ece(lrs, y, np.array([.5, .5]))

            self.assertEqual(context.exception.args[0], msg)

if __name__ == '__main__':
    unittest.main()
