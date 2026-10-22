import warnings
from typing import Any

import sklearn

from lir.transform import BinaryClassifierTransformer


class CalibratedSVC(BinaryClassifierTransformer):
    """
    Apply a calibrated :class:`~sklearn.svm.SVC` and output probabilities.

    The classifier is an :class:`~sklearn.svm.SVC` wrapped in a
    :class:`~sklearn.calibration.CalibratedClassifierCV` (with ``ensemble=False``),
    which replaces the deprecated use of ``SVC(probability=True)``.

    The input should be :class:`~lir.FeatureData`. The input for the model will be the features and the hypothesis
    labels. The output of ``apply()`` is a ``FeatureData`` object whose features are the probabilities of H1.

    Parameters
    ----------
    probability : bool, optional
        Deprecated parameter, accepted for backwards compatibility but ignored. The classifier provides calibrated
        probability estimates by default.
    **kwargs : dict
        Keyword arguments forwarded to :class:`~sklearn.svm.SVC`.
    """

    def __init__(self, probability: bool | None = None, **kwargs: Any):
        if probability is not None:
            warnings.warn(
                '`probability` is deprecated and will be removed in a future version. '
                'The SVM classifier is calibrated by default; remove `probability` from the configuration.',
                DeprecationWarning,
                stacklevel=2,
            )

        super().__init__(sklearn.calibration.CalibratedClassifierCV(sklearn.svm.SVC(**kwargs), ensemble=False))
