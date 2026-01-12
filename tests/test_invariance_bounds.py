import unittest

import numpy as np
from sklearn.linear_model import LogisticRegression

from lir.algorithms import invariance_bounds
from lir.algorithms.invariance_bounds import IVBounder
from lir.data.datasets.alcohol_breath_analyser import AlcoholBreathAnalyser
from lir.data.models import FeatureData, LLRData
from lir.transform.pipeline import Pipeline
from lir.util import Xn_to_Xy, probability_to_logodds
from tests.data.datasets.lr_bounding import BoundingExample4, BoundingExample5


class TestBounding(unittest.TestCase):
    def test_breath(self):
        llrs = AlcoholBreathAnalyser(ill_calibrated=True).get_instances()
        bounds = invariance_bounds.calculate_invariance_bounds(llrs)
        np.testing.assert_almost_equal(np.log10((0.1052741, 85.3731634)), bounds[:2], decimal=6)

    def test_iv_paper_examples(self):
        llr_threshold = np.arange(-2, 3, 0.001)

        llrs = BoundingExample4().get_instances()
        bounds = invariance_bounds.calculate_invariance_bounds(llrs, llr_threshold)
        np.testing.assert_almost_equal(np.log10((0.2382319, 2.7861212)), bounds[:2])

        llrs = BoundingExample5().get_instances()
        bounds = invariance_bounds.calculate_invariance_bounds(llrs, llr_threshold)
        np.testing.assert_almost_equal(np.log10((0.1412538, 38.1944271)), bounds[:2])

    def test_extreme_smallset(self):
        lrs = np.array([np.inf, 0])
        y = np.array([1, 0])
        data = LLRData(features=np.log10(lrs), labels=y)

        bounds = invariance_bounds.calculate_invariance_bounds(data)
        np.testing.assert_almost_equal((-0.477, 0.477), bounds[:2])

    def test_extreme(self):
        lrs = np.array([np.inf, np.inf, np.inf, 0, 0, 0])
        y = np.array([1, 1, 1, 0, 0, 0])
        data = LLRData(features=np.log10(lrs), labels=y)

        bounds = invariance_bounds.calculate_invariance_bounds(data)
        np.testing.assert_almost_equal((-0.845, 0.845), bounds[:2])

    def test_system01(self):
        lrs = np.array([0.01, 0.1, 1, 10, 100])

        bounds_bad = invariance_bounds.calculate_invariance_bounds(
            LLRData(features=np.log10(lrs), labels=np.array([1, 1, 1, 0, 0]))
        )
        bounds_good1 = invariance_bounds.calculate_invariance_bounds(
            LLRData(features=np.log10(lrs), labels=np.array([0, 0, 1, 1, 1]))
        )
        bounds_good2 = invariance_bounds.calculate_invariance_bounds(
            LLRData(features=np.log10(lrs), labels=np.array([0, 0, 0, 1, 1]))
        )

        np.testing.assert_almost_equal((0, 0), bounds_bad[:2])
        np.testing.assert_almost_equal((-0.823, 0.574), bounds_good1[:2])
        np.testing.assert_almost_equal((-0.574, 0.823), bounds_good2[:2])

    def test_neutral_smallset(self):
        lrs = np.array([1, 1])
        y = np.array([1, 0])
        data = LLRData(features=np.log10(lrs), labels=y)
        bounds = invariance_bounds.calculate_invariance_bounds(data)
        np.testing.assert_almost_equal((0, 0), bounds[:2])

    def test_bias(self):
        lrs = np.ones(10) * 10
        y = np.concatenate([np.ones(9), np.zeros(1)])
        data = LLRData(features=np.log10(lrs), labels=y)
        bounds = invariance_bounds.calculate_invariance_bounds(data)
        np.testing.assert_almost_equal((0, 0.102), bounds[:2])

        lrs = np.concatenate([np.ones(10) * 10, np.ones(1)])
        y = np.concatenate([np.ones(10), np.zeros(1)])
        data = LLRData(features=np.log10(lrs), labels=y)
        bounds = invariance_bounds.calculate_invariance_bounds(data)
        np.testing.assert_almost_equal((0, 0.581), bounds[:2])

        lrs = np.concatenate([np.ones(10) * 1000, np.ones(1) * 1.1])
        y = np.concatenate([np.ones(10), np.zeros(1)])
        data = LLRData(features=np.log10(lrs), labels=y)
        bounds = invariance_bounds.calculate_invariance_bounds(data)
        np.testing.assert_almost_equal((0, 0.581), bounds[:2])

    def test_bounded_calibrated_scorer(self):
        rng = np.random.default_rng(0)

        X0 = rng.normal(loc=-1, scale=1, size=(1000, 1))
        X1 = rng.normal(loc=+1, scale=1, size=(1000, 1))
        X, y = Xn_to_Xy(X0, X1)

        bounder = IVBounder()
        pipeline = Pipeline(
            [
                ('logit', LogisticRegression()),
                ('to_logodds', probability_to_logodds),
                ('iv', bounder),
            ]
        )
        pipeline.fit(FeatureData(features=X, labels=y))
        bounds = (bounder.lower_llr_bound, bounder.upper_llr_bound)
        np.testing.assert_almost_equal((-1.6170015, 2.1899985), bounds)


if __name__ == '__main__':
    unittest.main()
