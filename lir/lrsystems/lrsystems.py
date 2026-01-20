from abc import ABC, abstractmethod
from typing import Self

from lir import Transformer
from lir.data.models import InstanceData, LLRData


class LRSystem(Transformer, ABC):
    """General representation of an LR system."""

    def fit(self, instances: InstanceData) -> Self:
        """Fit the LR system on a set of features and corresponding labels.

        The number of labels must be equal to the number of instances.
        """
        return self

    @abstractmethod
    def apply(self, instances: InstanceData) -> LLRData:
        """Use the LR system to calculate the LLR data from the instances.

        Applies the LR system on a set of instances, optionally with corresponding labels, and returns a
        representation of the calculated LLR data through the `LLRData` tuple.
        """
        raise NotImplementedError
