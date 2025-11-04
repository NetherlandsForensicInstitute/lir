from abc import ABC, abstractmethod
from typing import Any

from lir.data.models import FeatureData, FeatureDataType, LLRData
from lir.transform import Transformer, as_transformer


class Pipeline(Transformer):
    """
    A pipeline of processing modules.

    A module may be a scikit-learn style transformer, estimator, or a LIR `Transformer`
    """

    def __init__(self, steps: list[tuple[str, Transformer | Any]]):
        """
        Constructor.

        :param steps: the steps of the pipeline as a list of (name, module) tuples.
        """
        self.steps = [(name, as_transformer(module)) for name, module in steps]

    def fit(self, instances: FeatureData) -> "Pipeline":
        for name, module in self.steps[:-1]:
            instances = module.fit_transform(instances)

        if len(self.steps) > 0:
            _, last_module = self.steps[-1]
            last_module.fit(instances)

        return self

    def transform(self, instances: FeatureDataType) -> FeatureDataType:
        for name, module in self.steps:
            instances = module.transform(instances)
        return instances

    def fit_transform(self, instances: FeatureData) -> FeatureData:
        for name, module in self.steps:
            instances = module.fit_transform(instances)
        return instances


class LRSystem(ABC):
    """General representation of an LR system."""

    def __init__(self, name: str):
        self.name = name
        self.parameters: dict[str, Any] = {}

    def fit(self, instances: FeatureData) -> "LRSystem":
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
