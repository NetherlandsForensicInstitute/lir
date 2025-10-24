import numpy as np

from lir.lrsystems.lrsystems import LRSystem, Pipeline, LLRData


class SpecificSourceSystem(LRSystem):
    """Representation of a specific source, feature based LR system.

    In this strategy, a set of instances - captured within the
    feature vector X - and a set of (ground-truth) labels are used to train and
    afterward calculate corresponding LLR's for given feature vectors.
    """

    def __init__(self, name: str, pipeline: Pipeline):
        super().__init__(name)
        self.pipeline = pipeline

    def fit(self, instances: np.ndarray, labels: np.ndarray, meta: np.ndarray) -> "LRSystem":
        self.pipeline.fit(instances, labels)
        return self

    def apply(self, instances: np.ndarray, labels: np.ndarray | None, meta: np.ndarray) -> LLRData:
        """
        Applies the specific source LR system on a set of instances, optionally with corresponding labels, and returns a
        representation of the calculated LLR data through the `LLRData` tuple.

        The returned set of LLRs has the same order as the set of input instances, and the returned labels are unchanged
        from the input labels.
        """
        llrs = self.pipeline.transform(instances)

        return LLRData(
            llrs=llrs,
            labels=labels,
            meta_data=meta,
        )
