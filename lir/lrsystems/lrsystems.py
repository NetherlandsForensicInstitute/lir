from abc import ABC, abstractmethod
from typing import Any, Mapping

from lir import transform
from lir.data.models import FeatureData, FeatureDataType, LLRData
from lir.transform import AdvancedTransformer


class Pipeline:
    """
    A pipeline of processing modules.

    A module may be a scikit-learn style transformer, estimator, or a LIR `Transformer`
    """

    def __init__(self, steps: list[tuple[str, Any]]):
        """
        Constructor.

        :param steps: the steps of the pipeline as a list of (name, module) tuples.
        """
        self.steps = steps

    def _set_values(self, values: Mapping[str, Any]) -> None:
        for _, module in self.steps:
            if isinstance(module, AdvancedTransformer):
                for key, value in values.items():
                    module.set_value(key, value)

    def fit(self, instances: FeatureData) -> "Pipeline":
        self._set_values(
            {
                transform.TRANSFORMER_PHASE_KEY: "train",
                transform.TRANSFORMER_LABELS_KEY: instances.labels,
            },
        )

        features = instances.features
        for name, module in self.steps[:-1]:
            features = module.fit_transform(features, instances.labels)

        if len(self.steps) > 0:
            _, last_module = self.steps[-1]
            last_module.fit(features)

        return self

    def transform(self, instances: FeatureDataType) -> FeatureDataType:
        self._set_values(
            {
                transform.TRANSFORMER_PHASE_KEY: "test",
                transform.TRANSFORMER_LABELS_KEY: instances.labels,
            },
        )
        features = instances.features
        for name, module in self.steps:
            features = module.transform(features)
        return instances.replace(features=features)

    def fit_transform(self, instances: FeatureDataType) -> FeatureDataType:
        features = instances.features
        for name, module in self.steps:
            features = module.fit_transform(features, instances.labels)
        return instances.replace(features=features)


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
