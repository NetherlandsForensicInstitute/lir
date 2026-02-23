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
        """Fit the LR system on a set of features and corresponding labels.

        The number of labels must be equal to the number of instances.
        """
        return self

    def _apply_pipeline_and_attach_scores(self, pipeline: Transformer, data: InstanceData) -> LLRData:
        """Apply pipeline with optional score capture and attach scores to result.

        Parameters
        ----------
        pipeline : Transformer
            The pipeline to apply
        data : InstanceData
            The data to process

        Returns
        -------
        LLRData
            The result with scores attached if configured
        """
        sources_for_plots = getattr(self, 'sources_for_plots', None)

        # Set sources_for_plots on pipeline if it has the property
        if hasattr(pipeline, 'sources_for_plots'):
            pipeline.sources_for_plots = sources_for_plots

        result = pipeline.apply(data)
        llr_data = result.replace_as(LLRData)

        # Extract captured data if available
        captured = getattr(pipeline, '_captured', None)

        if sources_for_plots and captured is not None:
            try:
                captured_name, captured_data = captured
                if hasattr(captured_data, 'features'):
                    scores = captured_data.features
                    data_dict = llr_data.model_dump()
                    data_dict[captured_name] = scores
                    llr_data = LLRData(**data_dict)
                else:
                    LOG.warning('Captured data has no features attribute')
            except Exception as e:
                LOG.warning(f'Could not extract scores: {e}')
        return llr_data

    @abstractmethod
    def apply(self, instances: InstanceData) -> LLRData:
        """Use the LR system to calculate the LLR data from the instances.

        Applies the LR system on a set of instances, optionally with corresponding labels, and returns a
        representation of the calculated LLR data through the `LLRData` tuple.
        """
        raise NotImplementedError
