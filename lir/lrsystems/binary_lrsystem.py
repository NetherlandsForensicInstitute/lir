from typing import Self

from lir.data.models import InstanceData
from lir.lrsystems.lrsystems import LLRData, LRSystem
from lir.lrsystems.score_based import Pipeline


class BinaryLRSystem(LRSystem):
    """
    LR system for binary data and a linear pipeline.

    This may be used in specific source feature based LR systems.

    In this strategy, a set of instances - captured within the
    feature vector X - and a set of (ground-truth) labels are used to train and
    afterward calculate corresponding LLR's for given feature vectors.

    Parameters
    ----------
    pipeline : Transformer
        Transformer pipeline used to fit and score instances.
    save_features_after_step : dict[str, str] | None
        Optional dictionary of step names to capture intermediate output from. The keys of the dictionary are the
        names of the fields in which to save the features, and the values are the names of the steps after which to save
        the features. If a value is 'STARTING_DATA', the features are saved before applying any steps.
    """

    def __init__(self, pipeline: Pipeline, save_features_after_step: dict[str, str] | None = None):
        super().__init__()
        self.pipeline = pipeline
        self.pipeline.save_features_after_step = save_features_after_step
        # Keep the reverse lookup in sync so requested intermediate features are actually captured.
        self.pipeline.save_features_after_step_reversed = (
            {v: k for k, v in save_features_after_step.items()} if save_features_after_step else {}
        )

    def fit(self, instances: InstanceData) -> Self:
        """
        Fit the model on the given instance data.

        Parameters
        ----------
        instances : InstanceData
            Input instances to be processed by this method.

        Returns
        -------
        Self
            This LR system instance after fitting the pipeline.
        """
        self.pipeline.fit(instances)
        return self

    def apply(self, instances: InstanceData) -> LLRData:
        """
        Use LR system to calculate the LLR data from the instance data.

        Applies the specific source LR system on a set of instances, optionally with corresponding labels, and returns a
        representation of the calculated LLR data through the `LLRData` tuple.

        The returned set of LLRs has the same order as the set of input instances, and the returned labels are unchanged
        from the input labels.

        Parameters
        ----------
        instances : InstanceData
            Input instances to be processed by this method.

        Returns
        -------
        LLRData
            Likelihood-ratio data produced by applying the LR system.
        """
        llrs = self.pipeline.apply(instances)
        return LLRData(**llrs.model_dump())
