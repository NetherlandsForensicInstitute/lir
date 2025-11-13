import os
import numpy as np

from lir.data.models import DataSet, LLRData


class UnboundLRs(DataSet):
    """"
    Examples from paper:
        A transparent method to determine limit values for Likelihood Ratio systems, by
        Ivo Alberink, Jeannette Leegwater, Jonas Malmborg, Anders Nordgaard, Marjan Sjerps, Leen van der Ham
        In: Submitted for publication in 2025.
    """

    def __init__(self, example: int = 4):
        self.example = example

    def get_instances(self) -> LLRData:
        if self.example == 4:
            np.random.seed(42)
            llrs_h1 = np.log10(np.exp(np.random.normal(1, 1, 100)))
            llrs_h2 = np.log10(np.exp(np.random.normal(0, 1, 1000)))
        elif self.example == 5:
            dirname = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            input_path = os.path.join(dirname, 'resources/lr_bounding')
            llrs_h1 = np.loadtxt(os.path.join(input_path, 'LLR_KM.csv'))
            llrs_h2 = np.loadtxt(os.path.join(input_path, 'LLR_KNM.csv'))
        else:
            raise ValueError('Only examples 4 and 5 are supported.')
        llrs = np.append(llrs_h1, llrs_h2)
        y = np.append(np.ones((len(llrs_h1), 1)), np.zeros((len(llrs_h2), 1)))
        return LLRData(features=llrs, labels=y)
