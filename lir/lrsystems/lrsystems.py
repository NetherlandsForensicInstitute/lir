import logging
from abc import ABC, abstractmethod
from typing import Self

from lir import Transformer
from lir.data.models import InstanceData, LLRData


LOG = logging.getLogger(__name__)


class LRSystem(Transformer, ABC):
    """General representation of an LR system."""

    def set_score_source(self, score_source: str | None) -> Self:
        """Configure where scores should be extracted from.

        Parameters
        ----------
        score_source : str, optional
            Pipeline step name to extract scores from (e.g., "scaler", "logistic_regression").

        Returns
        -------
        Self
            Returns self for method chaining.
        """
        self.score_source = score_source
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
        score_source = getattr(self, 'score_source', None)

        # Set capture_step on pipeline if it has the property
        if hasattr(pipeline, 'capture_step'):
            pipeline.capture_step = score_source

        result = pipeline.apply(data)
        llr_data = result.replace_as(LLRData)

        # Extract captured data if available
        captured = getattr(pipeline, '_captured', None)

        if score_source and captured is not None:
            try:
                if hasattr(captured, 'features'):
                    scores = captured.features
                    data_dict = llr_data.model_dump()
                    data_dict['scores'] = scores
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
