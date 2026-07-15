import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, Protocol, Self

import numpy as np

from lir.data.models import FeatureData, InstanceData, InstanceDataType
from lir.util import check_type


LOG = logging.getLogger(__name__)


class DataWriter(Protocol):
    """Representation of a data writer and necessary methods."""

    def writerow(self, row: Any) -> None:
        """
        Write row to output.

        Parameters
        ----------
        row : Any
            CSV row dictionary to parse.
        """


class SKLearnPipelineModule(Protocol):
    """Representation of the interface required for estimators by the scikit-learn `Pipeline`."""

    def fit(self, X: np.ndarray, y: np.ndarray | None) -> Self: ...  # noqa: D102
    def transform(self, X: np.ndarray) -> Any: ...  # noqa: D102
    def predict_proba(self, X: np.ndarray) -> Any: ...  # noqa: D102


class SklearnTransformerType(Protocol):
    """Representation of the interface required for transformers by the scikit-learn `Pipeline`."""

    def fit(self, features: np.ndarray, labels: np.ndarray | None) -> Self: ...  # noqa: D102
    def transform(self, features: np.ndarray) -> Any: ...  # noqa: D102
    def fit_transform(self, features: np.ndarray, labels: np.ndarray | None) -> np.ndarray: ...  # noqa: D102


class Transformer(ABC):
    """
    Transformer module which is compatible with the scikit-learn `Pipeline`.

    The transformer should provide a `transform()` method. Since transformers are not
    fitted to the data, the `fit()` simply returns the object it was called upon without
    side effects.
    """

    def fit(self, instances: InstanceData) -> Self:
        """
        Perform (optional) fitting of the instance data.

        Parameters
        ----------
        instances : InstanceData
            Input instances to be processed by this method.

        Returns
        -------
        Self
            This transformer instance after fitting.
        """
        return self

    @abstractmethod
    def apply(self, instances: InstanceData) -> InstanceData:
        """
        Convert the instance data based on the (optionally fitted) model.

        Parameters
        ----------
        instances : InstanceData
            Input instances to be processed by this method.

        Returns
        -------
        InstanceData
            Instance data object produced by this operation.
        """
        raise NotImplementedError

    def fit_apply(self, instances: InstanceData) -> InstanceData:
        """
        Combine call to `fit()` with directly following call to `apply()`.

        Parameters
        ----------
        instances : InstanceData
            Input instances to be processed by this method.

        Returns
        -------
        InstanceData
            Instance data object produced by this operation.
        """
        return self.fit(instances).apply(instances)


class Identity(Transformer):
    """
    Represent the Identity function of a transformer.

    When `apply()` is called on such a transformer, it simply returns the instances.
    """

    def apply(self, instances: InstanceDataType) -> InstanceDataType:
        """
        Simply provide the instances.

        Parameters
        ----------
        instances : InstanceDataType
            Input instances to be processed by this method.

        Returns
        -------
        InstanceDataType
            Instance data object produced by this operation.
        """
        return instances


class BinaryClassifierTransformer(Transformer):
    """
    Implementation of a binary class classifier as scikit-learn `Pipeline` step.

    Parameters
    ----------
    estimator : SKLearnPipelineModule
        Estimator used to produce transformed or scored outputs.
    """

    def __init__(self, estimator: SKLearnPipelineModule):
        self.estimator = estimator

    def fit(self, instances: InstanceData) -> Self:
        """
        Fit the model on the provided instances.

        Parameters
        ----------
        instances : InstanceData
            Input instances to be processed by this method.

        Returns
        -------
        Self
            This transformer instance after fitting.
        """
        instances = check_type(FeatureData, instances)
        self.estimator.fit(instances.features, instances.hypothesis)
        return self

    def apply(self, instances: InstanceData) -> FeatureData:
        """
        Convert instances by applying the fitted model.

        Parameters
        ----------
        instances : InstanceData
            Input instances to be processed by this method.

        Returns
        -------
        FeatureData
            Instance data object produced by this operation.
        """
        instances = check_type(FeatureData, instances)

        # get probabilities from the estimator
        probabilities = self.estimator.predict_proba(instances.features)[:, 1]
        # return a copy of `instances` with the `features` attribute replaced by the newly obtained probabilities
        return instances.replace(features=probabilities)


class SklearnTransformer(Transformer):
    """
    Implementation of a binary class classifier as scikit-learn `Pipeline` step.

    Parameters
    ----------
    transformer : SklearnTransformerType
        Transformer instance wrapped by this adapter.
    """

    def __init__(self, transformer: SklearnTransformerType):
        self.transformer = transformer

    def fit(self, instances: InstanceData) -> Self:
        """
        Fit the model on the provided instances.

        Parameters
        ----------
        instances : InstanceData
            Input instances to be processed by this method.

        Returns
        -------
        Self
            This transformer instance after fitting.
        """
        instances = check_type(FeatureData, instances)
        self.transformer.fit(instances.features, instances.hypothesis)
        return self

    def apply(self, instances: InstanceData) -> InstanceData:
        """
        Convert instances by applying the fitted model.

        Parameters
        ----------
        instances : InstanceData
            Input instances to be processed by this method.

        Returns
        -------
        InstanceData
            Instance data object produced by this operation.
        """
        instances = check_type(FeatureData, instances)
        return instances.replace_as(FeatureData, features=self.transformer.transform(instances.features))

    def fit_apply(self, instances: InstanceData) -> FeatureData:
        """
        Combine call to `.fit()` followed by `.apply()`.

        Parameters
        ----------
        instances : InstanceData
            Input instances to be processed by this method.

        Returns
        -------
        FeatureData
            FeatureData object parsed from the source.
        """
        instances = check_type(FeatureData, instances)
        return instances.replace_as(
            FeatureData, features=self.transformer.fit_transform(instances.features, instances.hypothesis)
        )


class FunctionTransformer(Transformer):
    """
    Implementation of a transformer function as scikit-learn `Pipeline` step.

    Parameters
    ----------
    func : Callable
        Callable used to transform input instances.
    """

    def __init__(self, func: Callable):
        self.func = func

    def apply(self, instances: InstanceData) -> FeatureData:
        """
        Call the custom defined function on the feature data instances and use output as features.

        Parameters
        ----------
        instances : InstanceData
            Input instances to be processed by this method.

        Returns
        -------
        FeatureData
            FeatureData object parsed from the source.
        """
        instances = check_type(FeatureData, instances)
        return instances.replace(features=self.func(instances.features))


class Tee(Transformer):
    """
    Implementation of a custom transformer allowing to perform two separate tasks on a given input.

    Parameters
    ----------
    transformers : list[Transformer]
        Collection of transformers applied in sequence or parallel.
    """

    def __init__(self, transformers: list[Transformer]):
        super().__init__()
        self.transformers = transformers

    def fit(self, instances: InstanceData) -> Self:
        """
        Delegate `fit()` to all specified transformers.

        Parameters
        ----------
        instances : InstanceData
            Input instances to be processed by this method.

        Returns
        -------
        Self
            This tee transformer instance after delegating fit.
        """
        for transformer in self.transformers:
            transformer.fit(instances)

        return self

    def apply(self, instances: InstanceData) -> InstanceData:
        """
        Delegate `apply()` to all specified transformers.

        Parameters
        ----------
        instances : InstanceData
            Input instances to be processed by this method.

        Returns
        -------
        InstanceData
            Instance data object produced by this operation.
        """
        for transformer in self.transformers:
            transformer.apply(instances)

        return instances


class TransformerWrapper(Transformer):
    """
    Base class for a transformer wrapper.

    This class is derived from `AdvancedTransformer` and has a default implementation of all functions
    by forwarding the call to the wrapped transformer. A subclass may add or change functionality
    by overriding functions.

    Parameters
    ----------
    wrapped_transformer : Transformer
        Value passed via ``wrapped_transformer``.
    """

    def __init__(self, wrapped_transformer: Transformer):
        super().__init__()
        self.wrapped_transformer = wrapped_transformer

    def fit(self, instances: InstanceData) -> Self:
        """
        Delegate calls to underlying wrapped transformer but return the Wrapper instance.

        Parameters
        ----------
        instances : InstanceData
            Input instances to be processed by this method.

        Returns
        -------
        Self
            This wrapper instance after delegating fit.
        """
        self.wrapped_transformer.fit(instances)
        return self

    def apply(self, instances: InstanceData) -> InstanceData:
        """
        Delegate calls to underlying wrapped transformer but return the Wrapper instance.

        Parameters
        ----------
        instances : InstanceData
            Input instances to be processed by this method.

        Returns
        -------
        InstanceData
            Instance data object produced by this operation.
        """
        return self.wrapped_transformer.apply(instances)


def as_transformer(transformer_like: Any) -> Transformer:
    """
    Provide a `Transformer` instance of the provided transformer like input.

    For any transformer-like object, wrap if necessary, and return a `Transformer`.

    The transformer-like object may be one of the following:
    - an instance of `Transformer`, which is returned as-is;
    - a scikit-learn style transformer which implements `transform()` and optionally `fit()` and/or `fit_transform()`;
    - a scikit-learn style estimator, which implements `fit()` and `predict_proba()`; or
    - a callable which takes an `np.ndarray` argument and returns another `np.ndarray`.

    Parameters
    ----------
    transformer_like : Any
        Object to convert to the internal Transformer interface.

    Returns
    -------
    Transformer
        Equivalent object adapted to the internal ``Transformer`` interface.
    """
    if isinstance(transformer_like, Transformer):
        # The component already supports all necessary methods, through the `Transformer` interface.
        return transformer_like
    if hasattr(transformer_like, 'transform'):
        # The component implements a `transform()` method, which means we assume it is a sklearn style transformer.
        return SklearnTransformer(transformer_like)
    if hasattr(transformer_like, 'predict_proba'):
        # The component has a `predict_proba` method, which means we assume it is a sklearn style estimator
        return BinaryClassifierTransformer(transformer_like)
    if callable(transformer_like):
        # The component is a function
        return FunctionTransformer(transformer_like)

    raise ValueError(f'unknown module type of class {type(transformer_like)}')
