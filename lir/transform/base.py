from abc import ABC, abstractmethod
from typing import Self

from lir import InstanceData


class Transformer(ABC):
    """
    Transformer module which is compatible with the scikit-learn `Pipeline`.

    The transformer should provide a `transform()` method. Since transformers are not
    fitted to the data, the `fit()` simply returns the object it was called upon without
    side effects.
    """

    def fit(self, instances: InstanceData) -> Self:
        """
        Perform (optional) fitting of the instance data.

        Parameters
        ----------
        instances : InstanceData
            Input instances to be processed by this method.

        Returns
        -------
        Self
            This transformer instance after fitting.
        """
        return self

    @abstractmethod
    def apply(self, instances: InstanceData) -> InstanceData:
        """
        Convert the instance data based on the (optionally fitted) model.

        Parameters
        ----------
        instances : InstanceData
            Input instances to be processed by this method.

        Returns
        -------
        InstanceData
            Instance data object produced by this operation.
        """
        raise NotImplementedError

    def fit_apply(self, instances: InstanceData) -> InstanceData:
        """
        Combine call to `fit()` with directly following call to `apply()`.

        Parameters
        ----------
        instances : InstanceData
            Input instances to be processed by this method.

        Returns
        -------
        InstanceData
            Instance data object produced by this operation.
        """
        return self.fit(instances).apply(instances)
