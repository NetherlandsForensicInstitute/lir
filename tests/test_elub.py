import numpy as np
import unittest
from sklearn.linear_model import LogisticRegression

from lir.algorithms import bayeserror
from lir.data.datasets.alcohol_breath_analyser import AlcoholBreathAnalyser
from lir.data.models import FeatureData
from lir.transform import BinaryClassifierTransformer, FunctionTransformer
from lir.transform.pipeline import Pipeline
from lir.util import Xn_to_Xy, probability_to_logodds, logodds_to_odds


class TestElub(unittest.TestCase):
    def test_breath(self):
        data = AlcoholBreathAnalyser(ill_calibrated=True).get_instances()
        bounds = bayeserror.elub(logodds_to_odds(data.llrs), data.labels, add_misleading=1)
        np.testing.assert_almost_equal((0.11051160265422605, 80.42823031359497), bounds)

    def test_extreme_smallset(self):
        lrs = np.array([np.inf, 0])
        y = np.array([1, 0])
        bounds = bayeserror.elub(lrs, y, add_misleading=1)
        np.testing.assert_almost_equal((1, 1), bounds)

    def test_extreme(self):
        lrs = np.array([np.inf, np.inf, np.inf, 0, 0, 0])
        y = np.array([1, 1, 1, 0, 0, 0])
        bounds = bayeserror.elub(lrs, y, add_misleading=1)
        np.testing.assert_almost_equal((0.3362165, 2.9742741), bounds)

    def test_system01(self):
        lrs = np.array([0.01, 0.1, 1, 10, 100])
        bounds_bad = bayeserror.elub(lrs, np.array([1, 1, 1, 0, 0]), add_misleading=1)
        bounds_good1 = bayeserror.elub(lrs, np.array([0, 0, 1, 1, 1]), add_misleading=1)
        bounds_good2 = bayeserror.elub(lrs, np.array([0, 0, 0, 1, 1]), add_misleading=1)

        np.testing.assert_almost_equal((1, 1), bounds_bad)
        np.testing.assert_almost_equal((0.3771282, 1.4990474), bounds_good1)
        np.testing.assert_almost_equal((0.6668633, 2.6507161), bounds_good2)

    def test_neutral_smallset(self):
        lrs = np.array([1, 1])
        y = np.array([1, 0])
        bounds = bayeserror.elub(lrs, y, add_misleading=1)
        np.testing.assert_almost_equal((1, 1), bounds)

    def test_bias(self):
        lrs = np.ones(10) * 10
        y = np.concatenate([np.ones(9), np.zeros(1)])
        np.testing.assert_almost_equal(
            (1, 1), bayeserror.elub(lrs, y, add_misleading=1)
        )

        lrs = np.concatenate([np.ones(10) * 10, np.ones(1)])
        y = np.concatenate([np.ones(10), np.zeros(1)])
        np.testing.assert_almost_equal(
            (1, 1.8039884), bayeserror.elub(lrs, y, add_misleading=1)
        )

        lrs = np.concatenate([np.ones(10) * 1000, np.ones(1) * 1.1])
        y = np.concatenate([np.ones(10), np.zeros(1)])
        np.testing.assert_almost_equal(
            (1, 1), bayeserror.elub(lrs, y, add_misleading=1)
        )

    def test_bounded_calibrated_scorer(self):
        rng = np.random.default_rng(seed=0)

        X0 = rng.normal(loc=-1, scale=1, size=(1000, 1))
        X1 = rng.normal(loc=+1, scale=1, size=(1000, 1))
        X, y = Xn_to_Xy(X0, X1)

        bounder = bayeserror.ELUBBounder()
        pipeline = Pipeline(
            [
                ("logit", BinaryClassifierTransformer(LogisticRegression())),
                ("to_logodds", FunctionTransformer(probability_to_logodds)),
                ("elub", bounder),
            ]
        )
        pipeline.fit(FeatureData(features=X, labels=y))
        bounds = bounder.lower_llr_bound, bounder.upper_llr_bound
        np.testing.assert_almost_equal((-1.5313757, 2.0211532), bounds)


if __name__ == "__main__":
    unittest.main()
