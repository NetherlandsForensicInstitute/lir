import numpy as np

from lir.data.models import DataSet, LLRData


class ExampleExternalDataset(DataSet):
    """An example dataset class defined in an external module."""

    def get_instances(self) -> LLRData:
        # Example implementation returning dummy data.
        return LLRData(features=np.zeros((2, 3)), labels=np.zeros((3,)))
