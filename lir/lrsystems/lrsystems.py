from abc import ABC, abstractmethod
from typing import Any, Mapping

import numpy as np
import sklearn.pipeline

from lir import transform
from lir.transform import AdvancedTransformer, Identity


class Pipeline(sklearn.pipeline.Pipeline):
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

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray | None = None,
        meta: np.ndarray | None = None,
        **fit_params: dict[str, Any],
    ) -> "Pipeline":
        self._set_values(
            {
                transform.TRANSFORMER_PHASE_KEY: "train",
                transform.TRANSFORMER_LABELS_KEY: labels,
                transform.TRANSFORMER_META_KEY: meta,
            },
        )
        super().fit(features, labels, **fit_params)
        return self

    def transform(
        self,
        features: np.ndarray,
        labels: np.ndarray | None = None,
        meta: np.ndarray | None = None,
    ) -> np.ndarray:
        self._set_values(
            {
                transform.TRANSFORMER_PHASE_KEY: "test",
                transform.TRANSFORMER_LABELS_KEY: labels,
                transform.TRANSFORMER_META_KEY: meta,
            },
        )
        return super().transform(features)

    def fit_transform(
        self,
        features: np.ndarray,
        labels: np.ndarray | None = None,
        meta: np.ndarray | None = None,
        **fit_params: dict[str, Any],
    ) -> np.ndarray:
        self.fit(features, labels, meta, **fit_params)
        return super().transform(features)


class LRSystem(ABC):
    """General representation of an LR system."""

    def __init__(self, name: str):
        self.name = name
        self.parameters: dict[str, Any] = {}

    def fit(
        self, instances: np.ndarray, labels: np.ndarray, meta: np.ndarray
    ) -> "LRSystem":
        """
        Fits the LR system on a set of features and corresponding labels.

        The number of labels must be equal to the number of instances.
        """
        return self

    @abstractmethod
    def apply(
        self, instances: np.ndarray, labels: np.ndarray | None, meta: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
        """
        Applies the LR system on a set of instances, optionally with corresponding labels, and returns a set of LRs and
        their labels.

        The return value is a tuple of two numpy arrays of the same size, containing LRs and labels respectively. The
        second array should be `None` if the input data are unlabeled.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return self.name
