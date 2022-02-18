import csv
import unittest
from lir.util import to_probability, to_log_odds, to_odds
import numpy as np
import lir
from lir.metrics import devpav
from four_parameter_logistic_calibrator import FourParameterLogisticCalibrator
from sklearn.linear_model import LogisticRegression


def read_data(path):
    with open(path, 'r') as file:
        r = csv.reader(file)
        next(r)
        data = np.array([float(value) for _, value in r])
    return to_probability(np.array(data))


class TestFourParameterLogisticCalibrator(unittest.TestCase):
    X_diff = read_data('data/LRsdifferentnormalLLRdistribmu_s=1N_ss=300.csv')
    X_same = read_data('data/LRssamenormalLLRdistribmu_s=1N_ss=300.csv')

    def test_compare_to_logistic(self):
        X = np.concatenate([self.X_diff, self.X_same])
        y = np.concatenate([np.zeros(len(self.X_diff)), np.ones(len(self.X_same))])
        four_pl_model = FourParameterLogisticCalibrator()
        four_pl_model.fit(X, y)

        logistic = LogisticRegression(penalty='none', class_weight='balanced')
        logistic.fit(to_log_odds(X[:, None]), y)
        logistic_coef = [logistic.coef_[0][0], logistic.intercept_[0]]
        np.testing.assert_almost_equal(four_pl_model.coef_, logistic_coef, decimal=5)

    def test_pl_1_is_0(self):
        X_same = np.concatenate([self.X_same, [0]])
        X_diff = np.concatenate([self.X_diff, [0]])
        y = np.concatenate([np.zeros(len(X_diff)), np.ones(len(X_same))])
        X = np.concatenate([X_diff, X_same])

        four_pl_model = FourParameterLogisticCalibrator()
        four_pl_model.fit(X, y)

        probs = four_pl_model.predict_proba(X)[:, 1]
        odds = (to_odds(probs))
        np.testing.assert_equal(devpav(odds, y), 0.12029952948152635)

    def test_pl_0_is_1(self):
        X_same = np.concatenate([self.X_same, [1, 1-10**-10]])
        X_diff = np.concatenate([self.X_diff, [1, 1-10**-10]])
        y = np.concatenate([np.zeros(len(X_diff)), np.ones(len(X_same))])
        X = np.concatenate([X_diff, X_same])

        four_pl_model = FourParameterLogisticCalibrator()
        four_pl_model.fit(X, y)

        probs = four_pl_model.predict_proba(X)[:, 1]
        odds = (to_odds(probs))
        np.testing.assert_equal(devpav(odds, y), 0.15273304557837525)

    def test_pl_0_is_1_and_pl_1_is_0(self):
        X_same = np.concatenate([self.X_same, [0, 10**-10, 1, 1-10**-10]])
        X_diff = np.concatenate([self.X_diff, [0, 10**-10, 1, 1-10**-10]])
        y = np.concatenate([np.zeros(len(X_diff)), np.ones(len(X_same))])
        X = np.concatenate([X_diff, X_same])

        four_pl_model = FourParameterLogisticCalibrator()
        four_pl_model.fit(X, y)

        probs = four_pl_model.predict_proba(X)[:,1]
        odds = (to_odds(probs))
        np.testing.assert_equal(devpav(odds, y), 0.10475112893952891)