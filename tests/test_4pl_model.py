import csv
import unittest
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from lir.algorithms.logistic_regression import FourParameterLogisticCalibrator
from lir.data.models import LLRData
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

    def get_instances(self):
        X, y = Xn_to_Xy(self.X_diff, self.X_same)
        X = probability_to_logodds(X)
        return LLRData(features=X.reshape(-1, 1), labels=y)

    def xtest_compare_to_logistic(self):
        instances = self.get_instances()

        four_pl_model = FourParameterLogisticCalibrator()
        four_pl_model.fit(instances)

        logistic = LogisticRegression(penalty=None, class_weight='balanced')
        logistic.fit(instances.features, instances.labels)
        logistic_coef = [logistic.coef_[0][0], logistic.intercept_[0]]
        np.testing.assert_allclose(four_pl_model.coef_, logistic_coef, rtol=1e-2)

    def test_pl_1_is_0(self):
        instances = self.get_instances()
        instances += LLRData(features=np.array([-np.inf, -np.inf]), labels=np.array([0, 1]))

        four_pl_model = FourParameterLogisticCalibrator()
        four_pl_model.fit(instances)
        logodds = four_pl_model.transform(instances)

        np.testing.assert_almost_equal(devpav(logodds.llrs, logodds.labels), 0.12029952948152635, decimal=5)

    def test_pl_0_is_1(self):
        instances = self.get_instances()
        extra_instances_p = np.array([1, 1 - 10**-10, 1, 1 - 10**-10]).reshape(-1, 1)
        instances += LLRData(features=probability_to_logodds(extra_instances_p), labels=np.array([1, 1, 0, 0]))

        four_pl_model = FourParameterLogisticCalibrator()
        four_pl_model.fit(instances)
        logodds = four_pl_model.transform(instances)

        np.testing.assert_almost_equal(devpav(logodds.llrs, logodds.labels), 0.15273304557837525, decimal=5)

    def test_pl_0_is_1_and_pl_1_is_0(self):
        instances = self.get_instances()
        extra_instances_p = np.array([0, 10**-10, 1, 1 - 10**-10, 0, 10**-10, 1, 1 - 10**-10]).reshape(-1, 1)
        instances += LLRData(
            features=probability_to_logodds(extra_instances_p), labels=np.array([1, 1, 1, 1, 0, 0, 0, 0])
        )

        four_pl_model = FourParameterLogisticCalibrator()
        four_pl_model.fit(instances)

        logodds = four_pl_model.transform(instances)
        np.testing.assert_almost_equal(devpav(logodds.llrs, logodds.labels), 0.10475112893952891, decimal=5)


if __name__ == '__main__':
    unittest.main()
