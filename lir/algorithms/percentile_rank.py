from collections.abc import Callable

import numpy as np
import sklearn
from scipy.interpolate import interp1d
from scipy.stats import rankdata


class PercentileRankTransformer(sklearn.base.TransformerMixin):
    """
    Compute the percentile rankings of a dataset, relative to another dataset.

    Rankings are in range [0, 1]. Handling ties: the maximum of the ranks that
    would have been assigned to all the tied values is assigned to each value.

    To compute the ranks of dataset ``Z`` relative to dataset ``X``, :meth:`fit`
    will create a ranking function for each feature using ``X``. :meth:`transform`
    then applies those per-feature ranking functions to ``Z``.

    Both :meth:`fit` and :meth:`transform` accept an array ``X`` with one row per
    instance, i.e. shape ``(n_samples, n_features)``. The number of features must
    match between :meth:`fit` and :meth:`transform`.

    If ``X`` contains paired measurements per instance (shape
    ``(n_samples, n_features, 2)``), ranking is fitted and applied independently to
    the first and second measurement in the pair.

    Parameters
    ----------
    X : numpy.ndarray
        Input data with shape ``(n_samples, n_features)`` or
        ``(n_samples, n_features, 2)``.

    Returns
    -------
    numpy.ndarray
        Percentile ranks with the same shape as the input passed to
        :meth:`transform`.
    """

    def __init__(self) -> None:
        self.rank_functions: list[Callable] | None = None

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> 'PercentileRankTransformer':
        """Fit the transformer model on the data."""
        X = X.reshape(X.shape[0], -1)
        ranks_X = rankdata(X, method='max', axis=0) / X.shape[0]
        self.rank_functions = [
            interp1d(X[:, i], ranks_X[:, i], bounds_error=False, fill_value=(0, 1)) for i in range(X.shape[1])
        ]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Use the fitted model to transform the input data."""
        assert self.rank_functions, 'transform() called before fit()'
        original_shape = X.shape
        X = X.reshape(X.shape[0], -1)
        assert X.shape[1] == len(self.rank_functions), (
            f'number of features {X.shape[1]} does not match '
            'the number of features {len(self.rank_functions)} used for fit()'
        )
        ranks = [self.rank_functions[i](X[:, i]) for i in range(X.shape[1])]
        return np.stack(ranks, axis=1).reshape(*original_shape)
