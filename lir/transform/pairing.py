from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from lir.data.models import PairedFeatureData, concatenate_instances
from lir.lrsystems.lrsystems import FeatureData


class PairingMethod(ABC):
    """
    Base class for pairing methods.

    A pairing method should implement the `pair()` function.
    """

    @abstractmethod
    def pair(
        self,
        instances: FeatureData,
        n_trace_instances: int = 1,
        n_ref_instances: int = 1,
    ) -> PairedFeatureData:
        """
        Takes instances as input, and returns pairs.

        A pair may be a pair of sources, with multiple instances per source.

        The returned features have dimensions `(p, i, ...)`` where the first dimension is the pairs, the second
        dimension is the instances, and subsequent dimensions are the features.
        If the input has labels, the returned labels are an array of source labels, one label per pair, where the labels
        are 0=different source, 1=same source.
        Any other attributes are combined into tuples.

        :param instances: An array of instance features, with one row per instance.
        :param n_trace_instances: Number of instances per trace.
        :param n_ref_instances: Number of instances per reference source.
        :return: instance pairs
        """
        raise NotImplementedError


class SourcePairing(PairingMethod):
    """
    Returns paired sources (i.e. classes) from an array of instance features, labels and meta data.

    While pairing at instance level results in pairs of instances, some same-source and some different-source, pairing
    at source level results in pairing of multiple instances of source A against multiple instances of source B, where A
    and B can be same-source or different-source.

    If the input has `n` instances with `f` features,
    - the parameter `n_trace_instances` is the number of trace instances in each pair;
    - the parameter `n_ref_instances` is the number of reference instances in each pair;
    - the parameter `instances` has dimensions `(n, f)`;
    - the parameter `labels` has dimensions `(n,)`;
    - the parameter `meta` has dimensions `(n, ...)`.

    If the output has `p` pairs, this function returns a tuple of:
    - an array of instances with dimensions `(p, n_trace_instances+n_ref_instances, f)`;
    - an array of labels with dimensions `(p,)`;
    - an array of meta data with dimensions `(p, n_trace_instances+n_ref_instances, ...)`.
    """

    def __init__(
        self,
        same_source_limit: int | None = None,
        different_source_limit: int | None = None,
        ratio_limit: int | None = None,
        seed: Any | int = None,
    ):
        self._ss_limit = same_source_limit
        self._ds_limit = different_source_limit
        self._ratio_limit = ratio_limit
        self.rng = np.random.default_rng(seed=seed)

    def _get_random_subset(self, size: int, instances: FeatureData) -> FeatureData | None:
        if len(instances) < size:
            return None  # not enough data to generate a sufficiently large subset

        idx = self.rng.choice(np.arange(len(instances)), size, replace=False)
        return instances[idx]

    def _construct_array(
        self,
        label_pairs: np.ndarray,
        instances: FeatureData,
        n_trace_instances: int,
        n_ref_instances: int,
    ) -> PairedFeatureData | None:
        result_features: list[FeatureData] = []
        for trace_label, ref_label in label_pairs:
            if trace_label == ref_label:
                # construct a same-source pair
                pair_instances = self._get_random_subset(
                    n_trace_instances + n_ref_instances,
                    instances[instances.labels == trace_label],
                )
                if pair_instances is not None:
                    pair_instances = pair_instances.replace(labels=None)
                    result_features.append(pair_instances.apply(np.expand_dims, axis=0).replace(labels=np.ones(1)))
            else:
                # construct a different-source pair
                trace_instances = self._get_random_subset(
                    n_trace_instances,
                    instances[instances.labels == trace_label],
                )
                ref_instances = self._get_random_subset(
                    n_ref_instances,
                    instances[instances.labels == ref_label],
                )
                if trace_instances and ref_instances:
                    pair_instances = (trace_instances + ref_instances).replace(labels=None)
                    result_features.append(pair_instances.apply(np.expand_dims, axis=0).replace(labels=np.zeros(1)))

        if result_features:
            paired_features = concatenate_instances(*result_features)
            return paired_features.replace_as(
                PairedFeatureData, n_trace_instances=n_trace_instances, n_ref_instances=n_ref_instances
            )
        else:
            # construct an empty set of pairs
            target_shape = (0, n_trace_instances + n_ref_instances) + instances.features.shape[1:]
            return PairedFeatureData(
                n_trace_instances=n_trace_instances,
                n_ref_instances=n_ref_instances,
                features=np.empty(target_shape),
                labels=np.array([]),
            )

    def pair(
        self,
        instances: FeatureData,
        n_trace_instances: int = 1,
        n_ref_instances: int = 1,
    ) -> PairedFeatureData:
        """
        Pairs sources.

        Takes an array of features, labels, and meta, all with one row per instance.
        Returns an array of features, labels, and meta, all with one row per pair.
        The second dimension of the returned features and meta indicates the instance.

        For example, if the input features array has dimensions (x, 5), and the arguments `n_trace_instances` and
        `n_ref_instances` are both 1, the output features array has dimensions (p, 1+1, 5), with x is the number of
        input instances and p is the number of output pairs.
        """
        unique_labels = np.unique(instances.labels)

        label_pairing = np.array(np.meshgrid(unique_labels, unique_labels)).T.reshape(
            -1, 2
        )  # generate all possible pairs
        rows_same = label_pairing[:, 0] == label_pairing[:, 1]
        rows_diff = label_pairing[:, 0] < label_pairing[:, 1]

        same_source_pairs = label_pairing[rows_same]
        different_source_pairs = label_pairing[rows_diff]

        # reduce the number of same source pairs, if necessary
        if self._ss_limit is not None and np.sum(rows_same) > self._ss_limit:
            same_source_pairs = self.rng.choice(same_source_pairs, self._ss_limit, replace=False)

        ss_paired_data = self._construct_array(
            same_source_pairs,
            instances,
            n_trace_instances,
            n_ref_instances,
        )

        # reduce the number of different source pairs, if necessary
        n_ds_pairs = min(
            x
            for x in [
                (len(ss_paired_data) * self._ratio_limit if ss_paired_data and self._ratio_limit else None),
                self._ds_limit,
                different_source_pairs.shape[0],
            ]
            if x is not None
        )

        if n_ds_pairs < different_source_pairs.shape[0]:
            different_source_pairs = self.rng.choice(different_source_pairs, n_ds_pairs, replace=False)

        ds_paired_data = self._construct_array(
            different_source_pairs,
            instances,
            n_trace_instances,
            n_ref_instances,
        )

        paired_data = [subset for subset in [ss_paired_data, ds_paired_data] if subset]
        if paired_data:
            return concatenate_instances(*paired_data)
        else:
            features = np.zeros((0, n_trace_instances + n_ref_instances) + instances.features.shape[1:])
            labels = np.zeros((0,))
            return PairedFeatureData(
                features=features,
                labels=labels,
                n_trace_instances=n_trace_instances,
                n_ref_instances=n_ref_instances,
            )


class InstancePairing(PairingMethod):
    def __init__(
        self,
        same_source_limit=None,
        different_source_limit=None,
        ratio_limit=None,
        seed=None,
    ):
        """
        Returns paired instances from an array of instance features, labels and meta data.

        If the input has `n` instances with `f` features,
        - the parameter `features` has dimensions `(n, f)`;
        - the parameter `labels` has dimensions `(n,)`;
        - the parameter `meta` has dimensions `(n, ...)`.

        If the output has `p` pairs, this function returns a tuple of:
        - an array of features with dimensions `(p, 2, f)`;
        - an array of labels with dimensions `(p,)`;
        - an array of meta data with dimensions `(p, 2, ...)`.

        Note that this transformer may cause performance problems with large datasets,
        even if the number of instances in the output is limited.

        :param same_source_limit: the maximum number of same source pairs (None = no limit)
        :param different_source_limit: the maximum number of different source pairs (None = no limit; 'balanced' =
            number of same source pairs)
        :param ratio_limit: maximum ratio between same source and different source pairs.
                Ratio = ds pairs / ss pairs. The number of ds pairs will not exceed ratio_limit * ss pairs.
                If both ratio and same_source_limit/different_source_limit are specified,
                the number of pairs is chosen such that the ratio_limit is preserved and
                the limit(s) are not exceeded, while taking as many pairs as possible within these constraints.
        :param seed: seed to make pairing reproducible
        """
        self._ss_limit = same_source_limit
        self._ds_limit = different_source_limit
        self._ratio_limit = ratio_limit
        self._seed = seed
        self.__rng = None

    @property
    def rng(self):
        if not self.__rng:
            self.__rng = np.random.default_rng(seed=self._seed)
        return self.__rng

    def pair(
        self,
        instances: FeatureData,
        n_trace_instances: int = 1,
        n_ref_instances: int = 1,
    ) -> PairedFeatureData:
        if n_trace_instances != 1:
            raise ValueError(f'invalid values for `n_trace_instances`; expected: 1; found: {n_trace_instances}')
        if n_ref_instances != 1:
            raise ValueError(f'invalid values for `n_ref_instances`; expected: 1; found: {n_ref_instances}')

        self.__rng = None
        pairing = np.array(np.meshgrid(np.arange(len(instances)), np.arange(len(instances)))).T.reshape(
            -1, 2
        )  # generate all possible pairs
        same_source = instances.labels[pairing[:, 0]] == instances.labels[pairing[:, 1]]

        rows_same = np.where((pairing[:, 0] < pairing[:, 1]) & same_source)[
            0
        ]  # pairs with different id and same source
        rows_diff = np.where((pairing[:, 0] < pairing[:, 1]) & ~same_source)[
            0
        ]  # pairs with different id and different source

        if self._ss_limit is not None and rows_same.size > self._ss_limit:
            rows_same = self.rng.choice(rows_same, self._ss_limit, replace=False)

        n_ds_pairs = min(
            x
            for x in [
                rows_same.size * self._ratio_limit if self._ratio_limit else None,
                self._ds_limit,
                rows_diff.size,
            ]
            if x is not None
        )

        if n_ds_pairs < rows_diff.size:
            rows_diff = self.rng.choice(rows_diff, n_ds_pairs, replace=False)

        pairing = np.concatenate([pairing[rows_same, :], pairing[rows_diff, :]])
        pair_labels = np.concatenate([np.ones(rows_same.size), np.zeros(rows_diff.size)])

        # combine features by adding an extra dimension
        paired_data = instances[pairing[:, 0]].combine(instances[pairing[:, 1]], np.stack, axis=1)

        # apply the new labels: 1=same_source versus 0=different_source
        return paired_data.replace_as(
            PairedFeatureData, labels=pair_labels, instance_indices=pairing, n_trace_instances=1, n_ref_instances=1
        )
