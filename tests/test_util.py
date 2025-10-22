import numpy as np
import pytest

from lir import util


@pytest.mark.parametrize("logodds,odds,probability", [
    (0, 1, .5),
    (np.inf, np.inf, 1),
    (-np.inf, 0, 0),
    ([0, np.inf, -np.inf], [1, np.inf, 0], [.5, 1, 0]),
])
def test_logodds_to_odds(logodds: list[float], odds: list[float], probability: list[float]):
    np.allclose(np.array(odds), util.logodds_to_odds(np.array(logodds)))
    np.allclose(np.array(odds), util.probability_to_odds(np.array(probability)))

    np.allclose(np.array(logodds), util.odds_to_logodds(np.array(logodds)))
    np.allclose(np.array(logodds), util.probability_to_logodds(np.array(probability)))

    np.allclose(np.array(probability), util.logodds_to_probability(np.array(logodds)))
    np.allclose(np.array(probability), util.odds_to_probability(np.array(probability)))

    np.allclose(np.array(odds), util.logodds_to_odds(util.probability_to_logodds(util.odds_to_probability(np.array(odds)))))
    np.allclose(np.array(odds), util.probability_to_odds(util.logodds_to_probability(util.odds_to_logodds(np.array(odds)))))
