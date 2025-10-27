from abc import ABC, abstractmethod
from typing import Any, Mapping, Annotated, Self

import numpy as np
from pydantic import ConfigDict, AfterValidator, model_validator, BaseModel

from lir import transform
from lir.transform import AdvancedTransformer, Identity


def _validate_labels(labels: np.ndarray | None) -> np.ndarray | None:
    if labels is not None:
        if not isinstance(labels, np.ndarray):
            raise ValueError(f"labels must be None or np.ndarray; found: {type(labels)}")
        if len(labels.shape) != 1:
            raise ValueError(f"labels must be 1-dimensional; shape: {labels.shape}")
        unique_labels = np.unique(labels)
        if np.any((unique_labels != 0) & (unique_labels != 1)):
            raise ValueError(f"label values must be 0 or 1; found: {unique_labels}")

    return labels


class InstanceData(BaseModel):
    """
    Base class for data on instances.
    """
    model_config = ConfigDict(frozen=True, extra='allow', arbitrary_types_allowed=True)

    labels: Annotated[np.ndarray | None, AfterValidator(_validate_labels)] = None


class FeatureData(InstanceData):
    features: np.ndarray

    @model_validator(mode='after')
    def validate(self) -> Self:
        if self.labels is not None and self.labels.shape[0] != self.features.shape[0]:
            raise ValueError(f"dimensions of labels and features do not match; {self.labels.shape[0]} != {self.features.shape[0]}")
        return self


class LLRData(FeatureData):
    """Representation of calculated LLR values.

    The tuple contains same size numpy arrays of LLR values, corresponding
    meta-data and optionally the corresponding interval values and labels.
    """
    @property
    def llrs(self) -> np.ndarray:
        if len(self.features.shape) == 1:
            return self.features
        else:
            return self.features[:, 0]

    @property
    def llr_intervals(self):
        if len(self.features.shape) == 2 and self.features.shape[1] == 3:
            return self.features[:, 1:]
        else:
            return None

    def validate(self) -> Self:
        super().validate()

        if len(self.features.shape) > 2:
            raise ValueError(f"features must have 1 or 2 dimensions; shape: {self.features.shape}")
        if len(self.features.shape) == 2 and self.features.shape[1] != 3 and self.features.shape[1] != 1:
            raise ValueError(f"features must be 1-dimensional or 2-dimensional with 1 or 3 columns; shape: {self.features.shape}")

        return self


class Pipeline:
    def __init__(self, steps: list[tuple[str, Any]], **kwargs: dict):
        # an sklearn pipeline cannot be empty --> create an empty pipeline by adding the identity transformer
        if len(steps) == 0:
            steps = [("id", Identity())]

        super().__init__(steps, **kwargs)

    def _set_values(self, values: Mapping[str, Any]) -> None:
        for _, module in self.steps:
            if isinstance(module, AdvancedTransformer):
                for key, value in values.items():
                    module.set_value(key, value)

    def fit(self, instances: FeatureData) -> "Pipeline":
        self._set_values(
            {
                transform.TRANSFORMER_PHASE_KEY: "train",
                transform.TRANSFORMER_LABELS_KEY: labels,
                transform.TRANSFORMER_META_KEY: meta,
            },
        )
        super().fit(features, labels, **fit_params)
        return self

    def transform(self, instances: FeatureData) -> FeatureData:
        self._set_values(
            {
                transform.TRANSFORMER_PHASE_KEY: "test",
                transform.TRANSFORMER_LABELS_KEY: labels,
                transform.TRANSFORMER_META_KEY: meta,
            },
        )
        return super().transform(instances)

    def fit_transform(self, instances: FeatureData) -> FeatureData:
        self.fit(instances)
        return self.transform(instances)


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
