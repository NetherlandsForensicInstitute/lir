from typing import Self

from lir import Transformer
from lir.data.models import InstanceData
from lir.lrsystems.lrsystems import LLRData, LRSystem
from lir.transform.pairing import PairingMethod
from lir.transform.pipeline import Pipeline


class ScoreBasedSystem(LRSystem):
    """Provide a representation of a common source, score-based LR system.

    In this strategy, it is possible to prepare the data within
    a `preprocessing_pipeline`, create corresponding pairs of instances using
    the `pairing_function` and subsequently calculate scores as well as transform
    these scores to LLR's in the final `evaluation_pipeline`.
    """

    def __init__(
        self,
        preprocessing_pipeline: Transformer | None,
        pairing_function: PairingMethod,
        evaluation_pipeline: Transformer | None,
    ):
        super().__init__()
        self.preprocessing_pipeline = preprocessing_pipeline or Pipeline([])
        self.pairing_function = pairing_function
        self.evaluation_pipeline = evaluation_pipeline or Pipeline([])

    def fit(self, instances: InstanceData) -> Self:
        """Fit the model on the instance data."""
        instances = self.preprocessing_pipeline.fit_apply(instances)
        pairs = self.pairing_function.pair(instances, 1, 1)
        self.evaluation_pipeline.fit(pairs)
        return self

    def apply(self, instances: InstanceData) -> LLRData:
        """Use LR system to calculate LLR data from the instances.

        Applies the score-based LR system on a set of instances, optionally with corresponding labels, and returns a
        representation of the calculated LLR data through the `LLRData` tuple.

        The system takes instances as input, and calculates LLRs for pairs of instances. That means that there is a 2-1
        relation between input and output data.
        """
        instances = self.preprocessing_pipeline.apply(instances)
        pairs = self.pairing_function.pair(instances, 1, 1)
        return self._apply_pipeline_and_attach_scores(self.evaluation_pipeline, pairs)
