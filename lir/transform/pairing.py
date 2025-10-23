from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import sklearn.base


class PairingMethod(ABC):
    """
    Base class for pairing methods.

    A pairing method should implement the `pair()` function.
    """

    @abstractmethod
    def pair(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        meta: np.ndarray,
        n_trace_instances: int = 1,
        n_ref_instances: int = 1,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Takes instances as input, and returns pairs.

        A pair may be a pair of sources, with multiple instances per source.

        Parameters
        ----------
        features : np.ndarray
            An array of instance features, with one row per instance.
        labels : np.ndarray
            An array of source labels, one label per instance, where each label is unique for a source.
        meta : np.ndarray
            An array of instance meta data.
        n_trace_instances : int
            Number of instances per trace.
        n_ref_instances : int
            Number of instances per reference source.

        Returns
        -------
        features : np.ndarray
            An array of features of dimensions `(p, i, ...)`` where the first dimension is the pairs, the second
            dimension is the instances, and subsequent dimensions are the features.
        labels : np.ndarray
            An array of source labels, one label per pair, where the labels are 0=different source, 1=same source.
        meta : np.ndarray
            An array of meta data of dimensions `(p, i, ...)`, analogous to the features array.
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

    def _get_subset(
        self, size: int, features: np.ndarray, meta: np.ndarray
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        if features.shape[0] < size:
            return None, None

        idx = self.rng.choice(np.arange(features.shape[0]), size, replace=False)
        return features[idx], meta[idx]

    def _construct_array(
        self,
        label_pairs: np.ndarray,
        features: np.ndarray,
        labels: np.ndarray,
        meta: np.ndarray,
        n_trace_instances: int,
        n_ref_instances: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        result_features: list[np.ndarray] = []
        result_meta: list[np.ndarray] = []
        result_labels = []
        for trace_label, ref_label in label_pairs:
            if trace_label == ref_label:
                pair_features, pair_meta = self._get_subset(
                    n_trace_instances + n_ref_instances,
                    features[labels == trace_label],
                    meta[labels == trace_label],
                )
                if pair_features is not None and pair_meta is not None:
                    result_features.append(pair_features)
                    result_labels.append(1)
                    result_meta.append(pair_meta)
            else:
                trace_features, trace_meta = self._get_subset(
                    n_trace_instances,
                    features[labels == trace_label],
                    meta[labels == trace_label],
                )
                ref_features, ref_meta = self._get_subset(
                    n_ref_instances,
                    features[labels == ref_label],
                    meta[labels == ref_label],
                )
                if trace_features is not None and ref_features is not None:
                    result_features.append(np.concatenate([trace_features, ref_features]))
                    result_labels.append(0)
                    result_meta.append(np.concatenate([trace_meta, ref_meta]))

        if not result_labels:
            return (
                np.ones((0, n_trace_instances + n_ref_instances) + features.shape[1:]),
                np.array([]),
                np.ones((0, n_trace_instances + n_ref_instances) + meta.shape[1:]),
            )

        return (
            np.stack(result_features, axis=0),
            np.array(result_labels),
            np.stack(result_meta, axis=0),
        )

    def pair(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        meta: np.ndarray,
        n_trace_instances: int = 1,
        n_ref_instances: int = 1,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Pairs sources.

        Takes an array of features, labels, and meta, all with one row per instance.
        Returns an array of features, labels, and meta, all with one row per pair.
        The second dimension of the returned features and meta indicates the instance.

        For example, if the input features array has dimensions (x, 5), and the arguments `n_trace_instances` and
        `n_ref_instances` are both 1, the output features array has dimensions (p, 1+1, 5), with x is the number of
        input instances and p is the number of output pairs.
        """
        assert features.shape[0] == labels.shape[0]
        assert features.shape[0] == meta.shape[0]

        unique_labels = np.unique(labels)

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

        ss_pair_features, ss_pair_labels, ss_pair_meta = self._construct_array(
            same_source_pairs,
            features,
            labels,
            meta,
            n_trace_instances,
            n_ref_instances,
        )

        # reduce the number of different source pairs, if necessary
        n_ds_pairs = min(
            x
            for x in [
                (ss_pair_labels.shape[0] * self._ratio_limit if self._ratio_limit else None),
                self._ds_limit,
                different_source_pairs.shape[0],
            ]
            if x is not None
        )

        if n_ds_pairs < different_source_pairs.shape[0]:
            different_source_pairs = self.rng.choice(different_source_pairs, n_ds_pairs, replace=False)

        ds_pair_features, ds_pair_labels, ds_pair_meta = self._construct_array(
            different_source_pairs,
            features,
            labels,
            meta,
            n_trace_instances,
            n_ref_instances,
        )

        pair_features = np.concatenate([ss_pair_features, ds_pair_features])
        pair_labels = np.concatenate([ss_pair_labels, ds_pair_labels])
        pair_meta = np.concatenate([ss_pair_meta, ds_pair_meta])

        return pair_features, pair_labels, pair_meta


class InstancePairing(sklearn.base.TransformerMixin):
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
        self.rng = np.random.default_rng(seed=seed)

    def fit(self, X):
        return self

    def _transform(self, X, y) -> tuple[np.ndarray[Any, Any], Any]:
        pairing = np.array(np.meshgrid(np.arange(X.shape[0]), np.arange(X.shape[0]))).T.reshape(
            -1, 2
        )  # generate all possible pairs
        same_source = y[pairing[:, 0]] == y[pairing[:, 1]]

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
        self.pairing = pairing

        X = np.stack([X[pairing[:, 0]], X[pairing[:, 1]]], axis=2)  # pair instances by adding another dimension
        y = np.concatenate(
            [np.ones(rows_same.size), np.zeros(rows_diff.size)]
        )  # apply the new labels: 1=same_source versus 0=different_source

        return X, y

    def pair(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        meta: np.ndarray,
        n_trace_instances: int = 1,
        n_ref_instances: int = 1,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert features.shape[0] == labels.shape[0]
        assert features.shape[0] == meta.shape[0]

        if n_trace_instances != 1:
            raise ValueError(f"invalid values for `n_trace_instances`; expected: 1; found: {n_trace_instances}")
        if n_ref_instances != 1:
            raise ValueError(f"invalid values for `n_ref_instances`; expected: 1; found: {n_ref_instances}")

        pair_features, pair_labels = self._transform(features, labels)
        pair_features = pair_features.transpose(0, 2, 1)

        # if meta is 1d, increase dimension to 2
        meta = meta.reshape(labels.size, -1)
        pair_meta = np.stack([meta[self.pairing[:, 0]], meta[self.pairing[:, 1]]], axis=1)

        return pair_features, pair_labels, pair_meta
