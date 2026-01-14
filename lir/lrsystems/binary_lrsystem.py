from typing import Self

from lir import Transformer
from lir.data.models import InstanceData
from lir.lrsystems.lrsystems import LLRData, LRSystem


class BinaryLRSystem(LRSystem):
    """
    LR system for binary data and a linear pipeline.

    This may be used in specific source feature based LR systems.

    In this strategy, a set of instances - captured within the
    feature vector X - and a set of (ground-truth) labels are used to train and
    afterward calculate corresponding LLR's for given feature vectors.
    """

    def __init__(self, pipeline: Transformer):
        self.pipeline = pipeline

    def fit(self, instances: InstanceData) -> Self:
        self.pipeline.fit(instances)
        return self

    def apply(self, instances: InstanceData) -> LLRData:
        """
        Applies the specific source LR system on a set of instances, optionally with corresponding labels, and returns a
        representation of the calculated LLR data through the `LLRData` tuple.

        The returned set of LLRs has the same order as the set of input instances, and the returned labels are unchanged
        from the input labels.
        """
        llrs = self.pipeline.apply(instances)
        return LLRData(**llrs.model_dump())
