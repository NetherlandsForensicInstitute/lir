import numpy as np
import sklearn.isotonic
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array, check_consistent_length

from lir.util import probability_to_logodds


class IsotonicRegression(sklearn.isotonic.IsotonicRegression):
    """
    Sklearn implementation IsotonicRegression throws an error when values are Inf or -Inf when in fact
    IsotonicRegression can handle infinite values. This wrapper around the sklearn implementation of IsotonicRegression
    prevents the error being thrown when Inf or -Inf values are provided.
    """

    def fit(self, X: ArrayLike, y: ArrayLike, sample_weight: ArrayLike | tuple | None = None) -> 'IsotonicRegression':
        """Fit the model using X, y as training data.

        Parameters
        ----------
        X : array-like of shape (n_samples,)
            Training data.

        y : array-like of shape (n_samples,)
            Training target.

        sample_weight : array-like of shape (n_samples,), default=None
            Weights. If set to None, all weights will be set to 1 (equal
            weights).

        Returns
        -------
        self : object
            Returns an instance of self.

        Notes
        -----
        X is stored for future use, as :meth:`transform` needs X to interpolate
        new input data.
        """
        check_params = dict(accept_sparse=False, ensure_2d=False, ensure_all_finite=False)  # noqa: C408
        X = check_array(X, dtype=[np.float64, np.float32], **check_params)
        y = check_array(y, dtype=X.dtype, **check_params)
        check_consistent_length(X, y, sample_weight)

        # Transform y by running the isotonic regression algorithm and
        # transform X accordingly.
        X, y = self._build_y(X, y, sample_weight)

        # It is necessary to store the non-redundant part of the training set
        # on the model to make it possible to support model persistence via
        # the pickle module as the object built by scipy.interp1d is not
        # picklable directly.
        self._necessary_X_, self._necessary_y_ = X, y

        # Build the interpolation function
        self._build_f(X, y)
        return self

    def transform(self, T: ArrayLike) -> np.ndarray:
        """Transform new data by linear interpolation

        Parameters
        ----------
        T : array-like of shape (n_samples,)
            Data to transform.

        Returns
        -------
        T_ : array, shape=(n_samples,)
            The transformed data
        """

        dtype = self._necessary_X_.dtype if hasattr(self, '_necessary_X_') else np.float64

        T = check_array(T, dtype=dtype, ensure_2d=False, ensure_all_finite=False)

        if len(T.shape) != 1:
            raise ValueError('Isotonic regression input should be a 1d array')

        # Handle the out_of_bounds argument by clipping if needed
        if self.out_of_bounds not in ['raise', 'nan', 'clip']:
            raise ValueError(
                f"The argument ``out_of_bounds`` must be in 'nan', 'clip', 'raise'; got {self.out_of_bounds}"
            )

        if self.out_of_bounds == 'clip':
            T = np.clip(T, self.X_min_, self.X_max_)

        res = self.f_(T)

        # on scipy 0.17, interp1d up-casts to float64, so we cast back
        res = res.astype(T.dtype)

        return res


class IsotonicCalibrator(BaseEstimator, TransformerMixin):
    """
    Calculates a likelihood ratio of a score value, provided it is from one of
    two distributions. Uses isotonic regression for interpolation.

    In contrast to `IsotonicRegression`, this class:
    - has an initialization argument that provides the option of adding misleading data points
    - outputs logodds instead of probabilities
    """

    def __init__(self, add_misleading: int = 0):
        """

        :param add_misleading:
        """
        self.add_misleading = add_misleading
        self._ir = IsotonicRegression(out_of_bounds='clip')

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'IsotonicCalibrator':
        assert np.all(np.unique(y) == np.arange(2)), 'y labels must be 0 and 1'

        # prevent extreme LRs
        if self.add_misleading > 0:
            X = np.concatenate(
                [
                    X,
                    np.ones(self.add_misleading) * (X.max() + 1),
                    np.ones(self.add_misleading) * (X.min() - 1),
                ]
            )
            y = np.concatenate([y, np.zeros(self.add_misleading), np.ones(self.add_misleading)])

        prior = np.sum(y) / y.size
        weight = y * (1 - prior) + (1 - y) * prior
        self._ir.fit(X, y, sample_weight=weight)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        self.p1 = self._ir.transform(X)
        self.p0 = 1 - self.p1
        return probability_to_logodds(self.p1)
