import numpy as np

from lir.lrsystems.lrsystems import LRSystem, Pipeline, LLRData
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

    def fit(self, features: np.ndarray, labels: np.ndarray, meta: np.ndarray) -> "LRSystem":
        features = self.preprocessing_pipeline.fit_transform(features, labels)

        pair_features, pair_labels, pair_meta = self.pairing_function.pair(features, labels, meta, 1, 1)
        pair_features = pair_features.transpose(0, 2, 1)

        self.evaluation_pipeline.fit(pair_features, pair_labels)

        return self

    def apply(self, features: np.ndarray, labels: np.ndarray | None, meta: np.ndarray) -> LLRData:
        """
        Applies the score-based LR system on a set of instances, optionally with corresponding labels, and returns a
        representation of the calculated LLR data through the `LLRData` tuple.

        The system takes instances as input, and calculates LLRs for pairs of instances. That means that there is a 2-1
        relation between input and output data.
        """
        if labels is None:
            raise ValueError("pairing requires labels")

        features = self.preprocessing_pipeline.transform(features)

        pair_features, pair_labels, pair_meta = self.pairing_function.pair(features, labels, meta, 1, 1)
        pair_features = pair_features.transpose(0, 2, 1)

        pair_llrs = self.evaluation_pipeline.transform(pair_features)

        return LLRData(
            llrs=pair_llrs,
            labels=pair_labels,
            meta_data=pair_meta,
        )
