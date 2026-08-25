import numpy as np

from lir import LLRData

from .average_llr_metric_weighted import calculate_weighted_cllr


def test_average_llr():
    assert calculate_weighted_cllr(LLRData(features=np.array([np.inf, 0]), hypothesis=np.array([1, 0])), 0, 1) == 0
    assert calculate_weighted_cllr(LLRData(features=np.array([np.inf, 0]), hypothesis=np.array([1, 0])), 1, 0) == 1
    assert calculate_weighted_cllr(LLRData(features=np.array([np.inf, 0]), hypothesis=np.array([1, 0])), 1, 1) == 0.5
