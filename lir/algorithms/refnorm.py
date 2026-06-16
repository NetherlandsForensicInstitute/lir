from typing import Self

import numpy as np

from lir import FeatureData, InstanceData, PairedFeatureData, Transformer
from lir.data.models import pair_instances
from lir.util import check_type


class ReferenceNormalization(Transformer):
    """
    Transform the scores of the instance pairs with reference normalization.

    For each pair, a score is calculated and normalized against a set of reference data.

    Parameters
    ----------
    scorer : Transformer
        The scorer used to find the scores.
    refnorm_dataset : InstanceData
        The dataset from which to select measurement pairs to perform the refnorm transformation.
    normalize_trace : bool, optional
        Perform normalization on the trace instances.
    normalize_ref : bool, optional
        Perform normalization on the reference instances.
    remove_same_source : bool, optional
        Avoid instances to be normalized using same-source pairs, by checking the `source_ids`.
    """

    def __init__(
        self,
        scorer: Transformer,
        refnorm_dataset: InstanceData,
        normalize_trace: bool = True,
        normalize_ref: bool = False,
        remove_same_source: bool = True,
    ):
        super().__init__()
        self.scorer = scorer
        self.refnorm_dataset = refnorm_dataset
        self.normalize_trace = normalize_trace
        self.normalize_ref = normalize_ref
        self.remove_same_source = remove_same_source

    def _calculate_scores(self, pairs: InstanceData) -> np.ndarray:
        if self.remove_same_source:
            pairs = pairs[pairs.labels == 0]
        return check_type(FeatureData, self.scorer.apply(pairs)).features.reshape(-1)

    def _calculate_normalized_score(self, pair: PairedFeatureData) -> float:
        target_pair_duplicated = pair[np.zeros(len(self.refnorm_dataset))]
        norm_sets = []
        if self.normalize_trace:
            norm_pairs = pair_instances(target_pair_duplicated.trace_instances, self.refnorm_dataset)
            norm_sets.append(self._calculate_scores(norm_pairs))
        if self.normalize_ref:
            norm_pairs = pair_instances(self.refnorm_dataset, target_pair_duplicated.ref_instances)
            norm_sets.append(self._calculate_scores(norm_pairs))

        score = self._calculate_scores(pair)
        norm = 0.0
        for norms in norm_sets:
            norm += (score - np.mean(norms)) / np.std(norms, ddof=1)
        norm /= len(norm_sets)
        return norm

    def fit(self, instances: InstanceData) -> Self:  # numpydoc ignore=PR01,RT01
        """Fit the scorer on the training set."""
        self.scorer.fit(instances)
        return self

    def apply[DataType: InstanceData](self, instances: DataType) -> InstanceData:  # numpydoc ignore=PR01,RT01
        """Score the instances and apply reference normalization."""
        pairs = check_type(PairedFeatureData, instances)
        scores = np.array([self._calculate_normalized_score(pairs[i]) for i in range(len(pairs))])

        return pairs.replace_as(FeatureData, features=scores.reshape(-1, 1))
