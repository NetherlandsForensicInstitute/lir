from lir.lrsystems.lrsystems import LRSystem, Pipeline, LLRData, FeatureData
from lir.transform.pairing import PairingMethod


class ScoreBasedSystem(LRSystem):
    """Representation of a common source, score-based LR system.

    In this strategy, it is possible to prepare the data within
    a `preprocessing_pipeline`, create corresponding pairs of instances using
    the `pairing_function` and subsequently calculate scores as well as transform
    these scores to LLR's in the final `evaluation_pipeline`.
    """

    def __init__(
        self,
        name: str,
        preprocessing_pipeline: Pipeline | None,
        pairing_function: PairingMethod,
        evaluation_pipeline: Pipeline | None,
    ):
        super().__init__(name)
        self.preprocessing_pipeline = preprocessing_pipeline or Pipeline([])
        self.pairing_function = pairing_function
        self.evaluation_pipeline = evaluation_pipeline or Pipeline([])

    def fit(self, instances: FeatureData) -> "LRSystem":
        instances = self.preprocessing_pipeline.fit_transform(instances)
        pairs = self.pairing_function.pair(instances, 1, 1)
        self.evaluation_pipeline.fit(pairs)
        return self

    def apply(self, instances: FeatureData) -> LLRData:
        """
        Applies the score-based LR system on a set of instances, optionally with corresponding labels, and returns a
        representation of the calculated LLR data through the `LLRData` tuple.

        The system takes instances as input, and calculates LLRs for pairs of instances. That means that there is a 2-1
        relation between input and output data.
        """
        if not instances.has_labels:
            raise ValueError("pairing requires labels")

        instances = self.preprocessing_pipeline.transform(instances)
        pairs = self.pairing_function.pair(instances, 1, 1)
        pair_llrs = self.evaluation_pipeline.transform(pairs)

        return LLRData(**pair_llrs.model_dump())
