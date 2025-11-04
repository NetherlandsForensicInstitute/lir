from abc import ABC, abstractmethod
from typing import Any

from lir.data.models import FeatureData, LLRData


class LRSystem(ABC):
    """General representation of an LR system."""

    def __init__(self, name: str):
        self.name = name
        self.parameters: dict[str, Any] = {}

    def fit(self, instances: FeatureData) -> 'LRSystem':
        """
        Fits the LR system on a set of features and corresponding labels.

        The number of labels must be equal to the number of instances.
        """
        return self

    @abstractmethod
    def apply(self, instances: FeatureData) -> LLRData:
        """
        Applies the LR system on a set of instances, optionally with corresponding labels, and returns a
        representation of the calculated LLR data through the `LLRData` tuple.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return self.name
