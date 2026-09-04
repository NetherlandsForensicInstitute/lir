import numpy as np

from lir import LLRData


def average_llr(llrdata: LLRData) -> float:
    return np.average(llrdata.llrs)
