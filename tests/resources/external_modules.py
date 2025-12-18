import numpy as np

from lir.data.models import DataProvider, DataStrategy, LLRData


class ExampleExternalData(DataStrategy, DataProvider):
    """An example DataStrategy and DataProvider class defined in an external module."""

    def get_instances(self) -> LLRData:
        return LLRData(features=np.zeros((3, 1)), labels=np.zeros(3))

    def __iter__(self):
        return None
