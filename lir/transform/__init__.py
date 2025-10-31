from collections.abc import Callable, Iterator
import csv
import logging
from abc import abstractmethod, ABC
from itertools import chain
from pathlib import Path
from typing import Any, Protocol

from typing_extensions import Self

import numpy as np

from lir.data.models import FeatureData

LOG = logging.getLogger(__name__)


class DataWriter(Protocol):
    """Representation of a data writer and necessary methods."""

    def writerow(self, row: Any) -> None:
        pass


class SKLearnPipelineModule(Protocol):
    """Representation of the interface required for estimators by the scikit-learn `Pipeline`."""

    def fit(self, X: np.ndarray, y: np.ndarray | None) -> Self: ...
    def transform(self, X: np.ndarray) -> Any: ...
    def predict_proba(self, X: np.ndarray) -> Any: ...


class SklearnTransformerType(Protocol):
    """Representation of the interface required for transformers by the scikit-learn `Pipeline`."""

    def fit(self, features: np.ndarray, labels: np.ndarray | None) -> Self: ...
    def transform(self, features: np.ndarray) -> Any: ...
    def fit_transform(self, features: np.ndarray, labels: np.ndarray | None) -> np.ndarray: ...


class Transformer(ABC):
    """Transformer module which is compatible with the scikit-learn `Pipeline`.

    The transformer should provide a `transform()` method. Since transformers are not
    fitted to the data, the `fit()` simply returns the object it was called upon without
    side effects.
    """

    def fit(self, instances: FeatureData) -> Self:
        return self

    @abstractmethod
    def transform(self, instances: FeatureData) -> Any:
        """Each transformer should implement a custom `transform()` method."""
        raise NotImplementedError

    def fit_transform(self, instances: FeatureData) -> FeatureData:
        return self.fit(instances).transform(instances)


class Identity(Transformer):
    def transform(self, features: Any) -> Any:
        return features


class BinaryClassifierTransformer(Transformer):
    """Implementation of a binary class classifier as scikit-learn `Pipeline` step."""

    def __init__(self, estimator: SKLearnPipelineModule):
        self.estimator = estimator

    def fit(self, instances: FeatureData) -> Self:
        self.estimator.fit(instances.features, instances.labels)
        return self

    def transform(self, instances: FeatureData) -> FeatureData:
        return instances.replace(features=self.estimator.predict_proba(instances.features)[:, 1])


class SklearnTransformer(Transformer):
    """Implementation of a binary class classifier as scikit-learn `Pipeline` step."""

    def __init__(self, transformer: SklearnTransformerType):
        self.transformer = transformer

    def fit(self, instances: FeatureData) -> Self:
        self.transformer.fit(instances.features, instances.labels)
        return self

    def transform(self, instances: FeatureData) -> FeatureData:
        return instances.replace(features=self.transformer.transform(instances.features))

    def fit_transform(self, instances: FeatureData) -> FeatureData:
        return instances.replace(features=self.transformer.fit_transform(instances.features, instances.labels))


class FunctionTransformer(Transformer):
    """Implementation of a transformer function as scikit-learn `Pipeline` step."""

    def __init__(self, func: Callable):
        self.func = func

    def transform(self, instances: FeatureData) -> FeatureData:
        return instances.replace(features=self.func(instances.features))


class Tee(Transformer):
    """Implementation of a custom transformer allowing to perform two separate tasks on a given input."""

    def __init__(self, transformers: list[Transformer]):
        super().__init__()
        self.transformers = transformers

    def fit(self, instances: FeatureData) -> Self:
        for transformer in self.transformers:
            transformer.fit(instances)

        return self

    def transform(self, instances: FeatureData) -> FeatureData:
        for transformer in self.transformers:
            transformer.transform(instances)

        return instances


class TransformerWrapper(Transformer):
    """Base class for a transformer wrapper.

    This class is derived from `AdvancedTransformer` and has a default implementation of all functions
    by forwarding the call to the wrapped transformer. A sub class may add or change functionality
    by overriding functions.
    """

    def __init__(self, wrapped_transformer: Transformer):
        super().__init__()
        self.wrapped_transformer = wrapped_transformer

    def fit(self, instances: FeatureData) -> Self:
        self.wrapped_transformer.fit(instances)
        return self

    def transform(self, instances: FeatureData) -> FeatureData:
        return self.wrapped_transformer.transform(instances)


class NumpyTransformer(TransformerWrapper):
    """Implementation of a transformer wrapper to handle writing numpy data to CSV."""

    def __init__(self, transformer: Transformer, header: list[str] | None, path: Path):
        super().__init__(transformer)
        self.csv_writer = CsvWriter(header=header, path=path)

    def transform(self, instances: FeatureData) -> FeatureData:
        result = super().transform(instances)
        self.csv_writer.transform(result)
        return result


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
        phase: str = "test",
    ):
        super().__init__()
        self.path = path
        if self.path is None:
            raise ValueError("missing argument: path")

        self.header_prefix = self.path.stem

        self.input_files = include
        self.header = header
        self.include_labels = include_labels
        self.include_meta = include_meta
        self.include_input = include_input
        self.include_batch = include_batch
        self.write_on_phase = phase
        self.n_batches = 0

    def _join_reader(self, filename: str) -> Iterator[list[str]]:
        with open(self.path.parent / filename, "r") as f:
            reader = csv.reader(f)
            yield from reader

    def _write_rows(self, writer: DataWriter, instances: FeatureData, write_header: bool) -> None:
        all_headers: list[str] = []
        all_data: list[np.ndarray] = []

        if self.include_labels and instances.labels is not None:
            all_headers.append("label")
            all_data.append(instances.labels.reshape(-1, 1))

        if self.include_meta and hasattr(instances, "meta"):
            meta = instances.meta
            if len(meta.shape) < 2:
                meta = meta.reshape(meta.shape[0], -1)
            all_headers.extend([f"{self.header_prefix}.meta{n}" for n in range(meta.shape[1])])
            all_data.append(meta)

        if self.include_batch:
            all_headers.append("batch")
            all_data.append(np.ones((len(instances), 1)) * self.n_batches)

        if self.include_input:
            features = instances.features
            if len(features.shape) == 1:
                features = features.reshape(-1, 1)
            elif len(features.shape) != 2:
                raise ValueError(f"{__name__}: unsupported shape: {features.shape}")

            if self.header is None:
                all_headers.extend([f"{self.header_prefix}{i}" for i in range(features.shape[1])])
            else:
                all_headers.extend(self.header)
            all_data.append(features)

        if self.input_files is not None:
            for filename in self.input_files:
                rows = self._join_reader(filename)
                all_headers.extend(next(rows))
                all_data.append(np.array(rows))

        if write_header:
            writer.writerow(all_headers)
        for row in zip(*all_data):
            writer.writerow(chain(*row))

    def fit_transform(self, instances: FeatureData) -> FeatureData:
        return instances

    def transform(self, instances: FeatureData) -> FeatureData:
        """Write numpy feature vector to CSV output file."""
        LOG.info(f"writing CSV file: {self.path}")
        self.path.parent.mkdir(exist_ok=True, parents=True)
        if self.path.exists():
            write_mode = "a"
            write_header = False
        else:
            write_mode = "w"
            write_header = True

        with open(self.path, write_mode) as f:
            writer = csv.writer(f)
            self._write_rows(writer, instances, write_header)

        self.n_batches += 1

        return instances


def as_transformer(transformer_like: Any) -> Transformer:
    if isinstance(transformer_like, Transformer):
        # The component already supports all necessary methods, through the `Transformer` interface.
        return transformer_like
    if hasattr(transformer_like, "transform"):
        # The component implements a `transform()` method, which means we assume it is a sklearn style transformer.
        return SklearnTransformer(transformer_like)
    if hasattr(transformer_like, "predict_proba"):
        # The component has a `predict_proba` method, which means we assume it is a sklearn style estimator
        return BinaryClassifierTransformer(transformer_like)
    if callable(transformer_like):
        # The component is a function
        return FunctionTransformer(transformer_like)

    raise ValueError(f"unknown module type of class {type(transformer_like)}")
