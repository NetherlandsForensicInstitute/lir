import numpy as np

from lir import LLRData
from lir.metrics import cllr


def test_cllr(benchmark):
    llr_data = LLRData(hypothesis=np.zeros(10_000), features=np.ones(10_000))
    benchmark(cllr, llr_data)
