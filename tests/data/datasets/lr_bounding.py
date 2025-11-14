import os
import numpy as np

from enum import Enum, auto

from lir.data.models import DataSet, LLRData


class BoundingExample(Enum):
    EXAMPLE_4 = auto()
    EXAMPLE_5 = auto()


class UnboundLRs(DataSet):
    """"
    Examples from paper:
        A transparent method to determine limit values for Likelihood Ratio systems, by
        Ivo Alberink, Jeannette Leegwater, Jonas Malmborg, Anders Nordgaard, Marjan Sjerps, Leen van der Ham
        In: Submitted for publication in 2025.
    """

    def __init__(self, example: BoundingExample):
        if not isinstance(example, BoundingExample):
            raise ValueError("`example` must be an instance of `BoundingExample` enum.")

        self.example = example

        self.logic = {
            BoundingExample.EXAMPLE_4: self._example_4,
            BoundingExample.EXAMPLE_5: self._example_5,
        }

    def _example_4(self) -> tuple[np.ndarray, np.ndarray]:
        np.random.seed(42)
        llrs_h1 = np.log10(np.exp(np.random.normal(1, 1, 100)))
        llrs_h2 = np.log10(np.exp(np.random.normal(0, 1, 1000)))

        return llrs_h1, llrs_h2

    def _example_5(self) -> tuple[np.ndarray, np.ndarray]:
        dirname = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        input_path = os.path.join(dirname, 'resources/lr_bounding')
        llrs_h1 = np.loadtxt(os.path.join(input_path, 'LLR_KM.csv'))
        llrs_h2 = np.loadtxt(os.path.join(input_path, 'LLR_KNM.csv'))

        return llrs_h1, llrs_h2

    def get_instances(self) -> LLRData:
        llrs_h1, llrs_h2 = self.logic[self.example]()

        llrs = np.append(llrs_h1, llrs_h2)
        y = np.append(np.ones((len(llrs_h1), 1)), np.zeros((len(llrs_h2), 1)))
        return LLRData(features=llrs, labels=y)
