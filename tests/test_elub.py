import unittest

import numpy as np
from sklearn.linear_model import LogisticRegression

from lir.algorithms import bayeserror
from lir.data.models import FeatureData, LLRData
from lir.datasets.alcohol_breath_analyser import AlcoholBreathAnalyser
from lir.transform import BinaryClassifierTransformer, FunctionTransformer
from lir.transform.pipeline import Pipeline
from lir.util import Xn_to_Xy, probability_to_logodds


class TestElub(unittest.TestCase):
    def test_breath(self):
        data = AlcoholBreathAnalyser(ill_calibrated=True).get_instances()
        bounds = bayeserror.elub(data, add_misleading=1)
        np.testing.assert_almost_equal((0.1122018, 79.4328235), np.pow(10, bounds))

    def test_extreme_smallset(self):
        lrs = np.array([np.inf, 0])
        y = np.array([1, 0])
        data = LLRData(features=np.log10(lrs), labels=y)

        bounds = bayeserror.elub(data, add_misleading=1)
        np.testing.assert_almost_equal((1, 1), np.pow(10, bounds))

    def test_extreme(self):
        lrs = np.array([np.inf, np.inf, np.inf, 0, 0, 0])
        y = np.array([1, 1, 1, 0, 0, 0])
        data = LLRData(features=np.log10(lrs), labels=y)

        bounds = bayeserror.elub(data, add_misleading=1)
        np.testing.assert_almost_equal((0.3388442, 2.9512092), np.pow(10, bounds))

    def test_system01(self):
        lrs = np.array([0.01, 0.1, 1, 10, 100])
        data_bad = LLRData(features=np.log10(lrs), labels=np.array([1, 1, 1, 0, 0]))
        data_good1 = LLRData(features=np.log10(lrs), labels=np.array([0, 0, 1, 1, 1]))
        data_good2 = LLRData(features=np.log10(lrs), labels=np.array([0, 0, 0, 1, 1]))

        bounds_bad = bayeserror.elub(data_bad, add_misleading=1)
        bounds_good1 = bayeserror.elub(data_good1, add_misleading=1)
        bounds_good2 = bayeserror.elub(data_good2, add_misleading=1)

        np.testing.assert_almost_equal((1, 1), np.pow(10, bounds_bad))
        np.testing.assert_almost_equal((0.3801894, 1.4791084), np.pow(10, bounds_good1))
        np.testing.assert_almost_equal((0.6760830, 2.6302680), np.pow(10, bounds_good2))

    def test_neutral_smallset(self):
        lrs = np.array([1, 1])
        y = np.array([1, 0])
        data = LLRData(features=np.log10(lrs), labels=y)

        bounds = bayeserror.elub(data, add_misleading=1)
        np.testing.assert_almost_equal((1, 1), np.pow(10, bounds))

    def test_bias(self):
        lrs1 = np.ones(10) * 10
        y1 = np.concatenate([np.ones(9), np.zeros(1)])
        data1 = LLRData(features=np.log10(lrs1), labels=y1)

        np.testing.assert_almost_equal((1, 1), np.pow(10, bayeserror.elub(data1, add_misleading=1)))

        lrs2 = np.concatenate([np.ones(10) * 10, np.ones(1)])
        y2 = np.concatenate([np.ones(10), np.zeros(1)])
        data2 = LLRData(features=np.log10(lrs2), labels=y2)
        np.testing.assert_almost_equal((1, 1.7782794), np.pow(10, bayeserror.elub(data2, add_misleading=1)))

        lrs3 = np.concatenate([np.ones(10) * 1000, np.ones(1) * 1.1])
        y3 = np.concatenate([np.ones(10), np.zeros(1)])
        data3 = LLRData(features=np.log10(lrs3), labels=y3)
        np.testing.assert_almost_equal((1, 1), np.pow(10, bayeserror.elub(data3, add_misleading=1)))

    def test_bounded_calibrated_scorer(self):
        rng = np.random.default_rng(seed=0)

        X0 = rng.normal(loc=-1, scale=1, size=(1000, 1))
        X1 = rng.normal(loc=+1, scale=1, size=(1000, 1))
        X, y = Xn_to_Xy(X0, X1)

        bounder = bayeserror.ELUBBounder()
        pipeline = Pipeline(
            [
                ('logit', BinaryClassifierTransformer(LogisticRegression())),
                ('to_logodds', FunctionTransformer(probability_to_logodds)),
                ('elub', bounder),
            ]
        )
        pipeline.fit(FeatureData(features=X, labels=y))
        bounds = bounder.lower_llr_bound, bounder.upper_llr_bound
        np.testing.assert_almost_equal((-1.53, 2.02), bounds)


if __name__ == '__main__':
    unittest.main()
