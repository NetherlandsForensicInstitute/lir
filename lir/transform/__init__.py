import csv
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
from itertools import chain
from pathlib import Path
from typing import Any, Protocol, Self

import numpy as np

from lir.data.models import FeatureData, InstanceData, InstanceDataType
from lir.util import check_type


LOG = logging.getLogger(__name__)


class DataWriter(Protocol):
    """Representation of a data writer and necessary methods."""

    def writerow(self, row: Any) -> None:
        """Write row to output."""
        pass


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
    """Transformer module which is compatible with the scikit-learn `Pipeline`.

    The transformer should provide a `transform()` method. Since transformers are not
    fitted to the data, the `fit()` simply returns the object it was called upon without
    side effects.
    """

    def fit(self, instances: InstanceData) -> Self:
        """Perform (optional) fitting of the instance data."""
        return self

    @abstractmethod
    def apply(self, instances: InstanceData) -> InstanceData:
        """Convert the instance data based on the (optionally fitted) model."""
        raise NotImplementedError

    def fit_apply(self, instances: InstanceData) -> InstanceData:
        """Combine call to `fit()` with directly following call to `apply()`."""
        return self.fit(instances).apply(instances)


class Identity(Transformer):
    """Represent the Identity function of a transformer.

    When `apply()` is called on such a transformer, it simply returns the instances.
    """

    def apply(self, instances: InstanceDataType) -> InstanceDataType:
        """Simply provide the instances."""
        return instances


class BinaryClassifierTransformer(Transformer):
    """Implementation of a binary class classifier as scikit-learn `Pipeline` step."""

    def __init__(self, estimator: SKLearnPipelineModule):
        self.estimator = estimator

    def fit(self, instances: InstanceData) -> Self:
        """Fit the model on the provided instances."""
        instances = check_type(FeatureData, instances)
        self.estimator.fit(instances.features, instances.labels)
        return self

    def apply(self, instances: InstanceData) -> InstanceData:
        """Convert instances by applying the fitted model."""
        instances = check_type(FeatureData, instances)

        # get probabilities from the estimator
        probabilities = self.estimator.predict_proba(instances.features)[:, 1]
        # return a copy of `instances` with the `features` attribute replaced by the newly obtained probabilities
        return instances.replace(features=probabilities)


class SklearnTransformer(Transformer):
    """Implementation of a binary class classifier as scikit-learn `Pipeline` step."""

    def __init__(self, transformer: SklearnTransformerType):
        self.transformer = transformer

    def fit(self, instances: InstanceData) -> Self:
        """Fit the model on the provided instances."""
        instances = check_type(FeatureData, instances)
        self.transformer.fit(instances.features, instances.labels)
        return self

    def apply(self, instances: InstanceData) -> InstanceData:
        """Convert instances by applying the fitted model."""
        instances = check_type(FeatureData, instances)
        return instances.replace_as(FeatureData, features=self.transformer.transform(instances.features))

    def fit_apply(self, instances: InstanceData) -> FeatureData:
        """Combine call to `.fit()` followed by `.apply()`."""
        instances = check_type(FeatureData, instances)
        return instances.replace_as(
            FeatureData, features=self.transformer.fit_transform(instances.features, instances.labels)
        )


class FunctionTransformer(Transformer):
    """Implementation of a transformer function as scikit-learn `Pipeline` step."""

    def __init__(self, func: Callable):
        self.func = func

    def apply(self, instances: InstanceData) -> FeatureData:
        """Call the custom defined function on the feature data instances and use output as features."""
        instances = check_type(FeatureData, instances)
        return instances.replace(features=self.func(instances.features))


class Tee(Transformer):
    """Implementation of a custom transformer allowing to perform two separate tasks on a given input."""

    def __init__(self, transformers: list[Transformer]):
        super().__init__()
        self.transformers = transformers

    def fit(self, instances: InstanceData) -> Self:
        """Delegate `fit()` to all specified transformers."""
        for transformer in self.transformers:
            transformer.fit(instances)

        return self

    def apply(self, instances: InstanceData) -> InstanceData:
        """Delegate `apply()` to all specified transformers."""
        for transformer in self.transformers:
            transformer.apply(instances)

        return instances


class TransformerWrapper(Transformer):
    """Base class for a transformer wrapper.

    This class is derived from `AdvancedTransformer` and has a default implementation of all functions
    by forwarding the call to the wrapped transformer. A subclass may add or change functionality
    by overriding functions.
    """

    def __init__(self, wrapped_transformer: Transformer):
        super().__init__()
        self.wrapped_transformer = wrapped_transformer

    def fit(self, instances: InstanceData) -> Self:
        """Delegate calls to underlying wrapped transformer but return the Wrapper instance."""
        self.wrapped_transformer.fit(instances)
        return self

    def apply(self, instances: InstanceData) -> InstanceData:
        """Delegate calls to underlying wrapped transformer but return the Wrapper instance."""
        return self.wrapped_transformer.apply(instances)


class NumpyTransformer(TransformerWrapper):
    """Implementation of a transformer wrapper."""

    def __init__(self, transformer: Transformer, header: list[str] | None):
        super().__init__(transformer)
        self.header = header

    def apply(self, instances: InstanceData) -> InstanceData:
        """Extend the instances with the desired header data, call base `apply`."""
        instances = super().apply(instances)
        if self.header:
            instances = instances.replace(header=self.header)
        return instances

    def fit_apply(self, instances: InstanceData) -> InstanceData:
        """Extend the instances with the desired header data, call base `fit_apply`."""
        instances = super().fit_apply(instances)
        if self.header:
            instances = instances.replace(header=self.header)
        return instances


class CsvWriter(Transformer):
    """Implementation of a transformation step in a scikit-learn Pipeline that writes to CSV.

    This might be used to obtain temporary or intermediate results for logging or debugging
    purposes.
    """

    def __init__(
        self,
        path: Path,
        include: list[str] | None = None,
        header: list[str] | None = None,
        include_labels: bool = False,
        include_meta: bool = False,
        include_input: bool = True,
        include_batch: bool = False,
    ):
        super().__init__()
        self.path = path
        if self.path is None:
            raise ValueError('missing argument: path')

        self.header_prefix = self.path.stem

        self.input_files = include
        self.header = header
        self.include_labels = include_labels
        self.include_meta = include_meta
        self.include_input = include_input
        self.include_batch = include_batch
        self.n_batches = 0

    def _join_reader(self, filename: str) -> Iterator[list[str]]:
        with open(self.path.parent / filename, 'r') as f:
            reader = csv.reader(f)
            yield from reader

    def _write_rows(self, writer: DataWriter, instances: FeatureData, write_header: bool) -> None:
        all_headers: list[str] = []
        all_data: list[np.ndarray] = []

        if self.include_labels and instances.labels is not None:
            all_headers.append('label')
            all_data.append(instances.labels.reshape(-1, 1))

        if self.include_meta and hasattr(instances, 'meta'):
            meta = instances.meta
            if len(meta.shape) < 2:
                meta = meta.reshape(meta.shape[0], -1)
            all_headers.extend([f'{self.header_prefix}.meta{n}' for n in range(meta.shape[1])])
            all_data.append(meta)

        if self.include_batch:
            all_headers.append('batch')
            all_data.append(np.ones((len(instances), 1)) * self.n_batches)

        if self.include_input:
            features = instances.features
            if len(features.shape) == 1:
                features = features.reshape(-1, 1)
            elif len(features.shape) != 2:
                raise ValueError(f'{__name__}: unsupported shape: {features.shape}')

            if self.header is None:
                all_headers.extend([f'{self.header_prefix}{i}' for i in range(features.shape[1])])
            else:
                all_headers.extend(self.header)
            all_data.append(features)

        if self.input_files is not None:
            for filename in self.input_files:
                rows = self._join_reader(filename)
                all_headers.extend(next(rows))
                all_data.append(np.array(list(rows)))

        if write_header:
            writer.writerow(all_headers)
        for row in zip(*all_data, strict=True):
            writer.writerow(chain(*row))

    def fit_apply(self, instances: InstanceDataType) -> InstanceDataType:
        """Provide required `fit_apply()` and return all instances.

        Since the CsvWriter is implemented as a step (Transformer) in the pipeline, it should support
        the `fit_apply` method which is called on all transformers of the pipeline.

        We don't need to actually fit or transform anything, so we simply return the instances (as is).
        """
        return instances

    def apply(self, instances: InstanceData) -> FeatureData:
        """Write numpy feature vector to CSV output file."""
        instances = check_type(FeatureData, instances)

        LOG.info(f'writing CSV file: {self.path}')
        self.path.parent.mkdir(exist_ok=True, parents=True)
        if self.path.exists():
            write_mode = 'a'
            write_header = False
        else:
            write_mode = 'w'
            write_header = True

        with open(self.path, write_mode, newline='') as f:
            writer = csv.writer(f)
            self._write_rows(writer, instances, write_header)

        self.n_batches += 1

        return instances


def as_transformer(transformer_like: Any) -> Transformer:
    """Provide a `Transformer` instance of the provided transformer like input.

    For any transformer-like object, wrap if necessary, and return a `Transformer`.

    The transformer-like object may be one of the following:
    - an instance of `Transformer`, which is returned as-is;
    - a scikit-learn style transformer which implements `transform()` and optionally `fit()` and/or `fit_transform()`;
    - a scikit-learn style estimator, which implements `fit()` and `predict_proba()`; or
    - a callable which takes an `np.ndarray` argument and returns another `np.ndarray`.

    :param transformer_like: the object to wrap or return
    :return: a `Transformer` instance
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
