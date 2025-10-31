import numpy as np
import sklearn


class AbsDiffTransformer(sklearn.base.TransformerMixin):
    """
    Takes an array of sample pairs and returns the element-wise absolute difference.

    Expects:
        - X is of shape (n,f,2) with n=number of pairs; f=number of features; 2=number of samples per pair;
    Returns:
        - X has shape (n, f)
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert len(X.shape) == 3
        assert X.shape[1] == 2

        return np.abs(X[:, 0, :] - X[:, 1, :])
