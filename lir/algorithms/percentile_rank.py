from typing import Optional, Callable

import numpy as np
import sklearn
from scipy.interpolate import interp1d
from scipy.stats import rankdata


class PercentileRankTransformer(sklearn.base.TransformerMixin):
    """
    Compute the percentile rankings of a dataset, relative to another dataset.
    Rankings are in range [0, 1]. Handling ties: the maximum of the ranks that
    would have been assigned to all the tied values is assigned to each value.

    To be able to compute the rankings of dataset *Z* relative to dataset *X*,
    `fit` will create a ranking function for each feature separately, based on
    *X*. The method `transform` will apply ranking of *Z* based on dataset *X*.

    This class has the methods `fit()` and `transform()`, both take a parameter
    `X` with one row per instance, e.g. dimensions (n, f) with n = number of
    measurements, f = number of features. The number of features should be the
    same in `fit()` and `transform()`.

    If the parameter `X` has a *pair* of measurements per row, i.e. has
    dimensions (n, f, 2), the percentile rank is fitted and applied
    independently for the first and second measurement of the pair.

    Fit:
    Expects:
        - `X` is a numpy array with one row per instance

    Transform:
    Expects:
        - `X` is a numpy array with one row per instance
    Returns:
        - a numpy array with the same shape as `X`
    """

    def __init__(self) -> None:
        self.rank_functions: list[Callable] | None = None

    def fit(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> "PercentileRankTransformer":
        X = X.reshape(X.shape[0], -1)
        ranks_X = rankdata(X, method="max", axis=0) / X.shape[0]
        self.rank_functions = [
            interp1d(X[:, i], ranks_X[:, i], bounds_error=False, fill_value=(0, 1))
            for i in range(X.shape[1])
        ]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.rank_functions, "transform() called before fit()"
        original_shape = X.shape
        X = X.reshape(X.shape[0], -1)
        assert X.shape[1] == len(self.rank_functions), (
            f"number of features {X.shape[1]} does not match "
            "the number of features {len(self.rank_functions)} used for fit()"
        )
        ranks = [self.rank_functions[i](X[:, i]) for i in range(X.shape[1])]
        return np.stack(ranks, axis=1).reshape(*original_shape)
