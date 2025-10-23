import numpy as np
import unittest
from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline

from lir.algorithms import invariance_bounds
from lir.algorithms.invariance_bounds import IVBounder
from lir.data.datasets.alcohol_breath_analyser import AlcoholBreathAnalyser
from lir.transform import BinaryClassifierTransformer, FunctionTransformer
from lir.util import Xn_to_Xy, probability_to_logodds, logodds_to_odds


class TestBounding(unittest.TestCase):
    def test_breath(self):
        llrs, y, meta = AlcoholBreathAnalyser(ill_calibrated=True).get_instances()
        bounds = invariance_bounds.calculate_invariance_bounds(logodds_to_odds(llrs), y)
        np.testing.assert_almost_equal((0.1052741, 85.3731634), bounds[:2])

    def test_extreme_smallset(self):
        lrs = np.array([np.inf, 0])
        y = np.array([1, 0])
        bounds = invariance_bounds.calculate_invariance_bounds(lrs, y)
        np.testing.assert_almost_equal((0.3335112, 2.9999248), bounds[:2])

    def test_extreme(self):
        lrs = np.array([np.inf, np.inf, np.inf, 0, 0, 0])
        y = np.array([1, 1, 1, 0, 0, 0])
        bounds = invariance_bounds.calculate_invariance_bounds(lrs, y)
        np.testing.assert_almost_equal((0.1429257, 6.9840986), bounds[:2])

    def test_system01(self):
        lrs = np.array([0.01, 0.1, 1, 10, 100])
        bounds_bad = invariance_bounds.calculate_invariance_bounds(
            lrs, np.array([1, 1, 1, 0, 0])
        )
        bounds_good1 = invariance_bounds.calculate_invariance_bounds(
            lrs, np.array([0, 0, 1, 1, 1])
        )
        bounds_good2 = invariance_bounds.calculate_invariance_bounds(
            lrs, np.array([0, 0, 0, 1, 1])
        )

        np.testing.assert_almost_equal((1, 1), bounds_bad[:2])
        np.testing.assert_almost_equal((0.1503142, 3.74973), bounds_good1[:2])
        np.testing.assert_almost_equal((0.2666859, 6.6527316), bounds_good2[:2])

    def test_neutral_smallset(self):
        lrs = np.array([1, 1])
        y = np.array([1, 0])
        bounds = invariance_bounds.calculate_invariance_bounds(lrs, y)
        np.testing.assert_almost_equal((1, 1), bounds[:2])

    def test_bias(self):
        lrs = np.ones(10) * 10
        y = np.concatenate([np.ones(9), np.zeros(1)])
        bounds = invariance_bounds.calculate_invariance_bounds(lrs, y)
        np.testing.assert_almost_equal((1, 1.2647363), bounds[:2])

        lrs = np.concatenate([np.ones(10) * 10, np.ones(1)])
        y = np.concatenate([np.ones(10), np.zeros(1)])
        bounds = invariance_bounds.calculate_invariance_bounds(lrs, y)
        np.testing.assert_almost_equal((1, 3.8106582), bounds[:2])

        lrs = np.concatenate([np.ones(10) * 1000, np.ones(1) * 1.1])
        y = np.concatenate([np.ones(10), np.zeros(1)])
        bounds = invariance_bounds.calculate_invariance_bounds(lrs, y)
        np.testing.assert_almost_equal((1, 3.8106582), bounds[:2])

    def test_bounded_calibrated_scorer(self):
        rng = np.random.default_rng(0)

        X0 = rng.normal(loc=-1, scale=1, size=(1000, 1))
        X1 = rng.normal(loc=+1, scale=1, size=(1000, 1))
        X, y = Xn_to_Xy(X0, X1)

        bounder = IVBounder()
        pipeline = Pipeline(
            [
                ("logit", BinaryClassifierTransformer(LogisticRegression())),
                ("to_logodds", FunctionTransformer(probability_to_logodds)),
                ("iv", bounder),
            ]
        )
        pipeline.fit(X, y)
        bounds = (bounder.lower_llr_bound, bounder.upper_llr_bound)
        np.testing.assert_almost_equal((-1.6170015, 2.1899985), bounds)


if __name__ == "__main__":
    unittest.main()
