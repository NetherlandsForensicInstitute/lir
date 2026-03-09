import logging
from abc import ABC, abstractmethod
from typing import Self

from lir import Transformer
from lir.data.models import InstanceData, LLRData


LOG = logging.getLogger(__name__)


class LRSystem(Transformer, ABC):
    """General representation of an LR system."""

    def set_sources_for_plots(self, sources_for_plots: dict[str, str] | None) -> Self:
        """Configure where scores should be extracted from.

        Parameters
        ----------
        sources_for_plots : dict, optional
            Pipeline step name to extract scores from (e.g., "scaler", "logistic_regression").

        Returns
        -------
        Self
            Returns self for method chaining.
        """
        self.sources_for_plots = sources_for_plots
        return self

    def fit(self, instances: InstanceData) -> Self:
        """
        Fit the LR system on a set of features and corresponding labels.

        The number of labels must be equal to the number of instances.

        Parameters
        ----------
        instances : InstanceData
            Input instances to be processed by this method.

        Returns
        -------
        Self
            This LR system instance after optional fitting.
        """
        return self

    @abstractmethod
    def apply(self, instances: InstanceData) -> LLRData:
        """
        Use the LR system to calculate the LLR data from the instances.

        Applies the LR system on a set of instances, optionally with corresponding labels, and returns a
        representation of the calculated LLR data through the `LLRData` tuple.

        Parameters
        ----------
        instances : InstanceData
            Input instances to be processed by this method.

        Returns
        -------
        LLRData
            Likelihood-ratio data produced by applying the LR system.
        """
        raise NotImplementedError
