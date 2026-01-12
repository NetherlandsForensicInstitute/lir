import csv
import unittest
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from lir.algorithms.logistic_regression import FourParameterLogisticCalibrator
from lir.metrics.devpav import devpav
from lir.util import Xn_to_Xy, odds_to_probability, probability_to_logodds


def read_data(path):
    with open(path, 'r') as file:
        r = csv.reader(file)
        next(r)
        data = np.array([float(value) for _, value in r])
    return odds_to_probability(np.array(data))


class TestFourParameterLogisticCalibrator(unittest.TestCase):
    dirname = Path(__file__).parent
    X_diff = read_data(dirname / 'resources/LRsdifferentnormalLLRdistribmu_s=1N_ss=300.csv')
    X_same = read_data(dirname / 'resources/LRssamenormalLLRdistribmu_s=1N_ss=300.csv')

    def test_compare_to_logistic(self):
        X, y = Xn_to_Xy(self.X_diff, self.X_same)
        X = probability_to_logodds(X)
        four_pl_model = FourParameterLogisticCalibrator()
        four_pl_model.fit(X, y)

        logistic = LogisticRegression(penalty=None, class_weight='balanced')
        logistic.fit(X[:, None], y)
        logistic_coef = [logistic.coef_[0][0], logistic.intercept_[0]]
        np.testing.assert_allclose(four_pl_model.coef_, logistic_coef, rtol=1e-2)

    def test_pl_1_is_0(self):
        X_same = np.concatenate([self.X_same, [0]])
        X_diff = np.concatenate([self.X_diff, [0]])
        X, y = Xn_to_Xy(X_diff, X_same)
        X = probability_to_logodds(X)

        four_pl_model = FourParameterLogisticCalibrator()
        four_pl_model.fit(X, y)

        logodds = four_pl_model.transform(X)
        np.testing.assert_almost_equal(devpav(logodds, y), 0.12029952948152635, decimal=5)

    def test_pl_0_is_1(self):
        X_same = np.concatenate([self.X_same, [1, 1 - 10**-10]])
        X_diff = np.concatenate([self.X_diff, [1, 1 - 10**-10]])
        X, y = Xn_to_Xy(X_diff, X_same)
        X = probability_to_logodds(X)

        four_pl_model = FourParameterLogisticCalibrator()
        four_pl_model.fit(X, y)

        logodds = four_pl_model.transform(X)
        np.testing.assert_almost_equal(devpav(logodds, y), 0.15273304557837525, decimal=5)

    def test_pl_0_is_1_and_pl_1_is_0(self):
        X_same = np.concatenate([self.X_same, [0, 10**-10, 1, 1 - 10**-10]])
        X_diff = np.concatenate([self.X_diff, [0, 10**-10, 1, 1 - 10**-10]])
        X, y = Xn_to_Xy(X_diff, X_same)
        X = probability_to_logodds(X)

        four_pl_model = FourParameterLogisticCalibrator()
        four_pl_model.fit(X, y)

        logodds = four_pl_model.transform(X)
        np.testing.assert_almost_equal(devpav(logodds, y), 0.10475112893952891, decimal=5)


if __name__ == '__main__':
    unittest.main()
