from collections.abc import Callable, Iterator, Sequence
import csv
import logging
from abc import abstractmethod, ABC
from itertools import chain
from pathlib import Path
from typing import Any, Protocol
from typing_extensions import Self

import numpy as np
from sklearn.base import TransformerMixin


LOG = logging.getLogger(__name__)

TRANSFORMER_LABELS_KEY = "labels"
TRANSFORMER_PHASE_KEY = "phase"
TRANSFORMER_META_KEY = "meta"


class DataWriter(Protocol):
    """Representation of a data writer and necessary methods."""

    def writerow(self, row: Any) -> None:
        pass


class SKLearnPipelineModule(Protocol):
    """Representation of the interface required for estimators by the scikit-learn `Pipeline`."""

    def fit(self, X: Sequence[Any], y: Sequence[Any] | None) -> Self: ...
    def transform(self, X: Any) -> Any: ...
    def predict_proba(self, X: Any) -> Any: ...


class Transformer(ABC, TransformerMixin):
    """Transformer module which is compatible with the scikit-learn `Pipeline`.

    The transformer should provide a `transform()` method. Since transformers are not
    fitted to the data, the `fit()` simply returns the object it was called upon without
    side effects.
    """

    def fit(self, X: Sequence[Any], y: Sequence[Any] | None = None) -> "Transformer":
        return self

    @abstractmethod
    def transform(self, X: Any) -> Any:
        """Each transformer should implement a custom `transform()` method."""
        raise NotImplementedError


class AdvancedTransformer(Transformer, ABC):
    """Extended Transformer class which keeps internal state to be more flexible.

    This type of transformer can be used if another "side effect" is desired while
    performing a certain `Pipeline` step.
    """

    def __init__(self) -> None:
        self._values: dict[str, Any] = {}

    def set_value(self, key: str, value: Any | None) -> "AdvancedTransformer":
        self._values[key] = value
        return self


class Identity(Transformer):
    def transform(self, features: Any) -> Any:
        return features


class BinaryClassifierTransformer(Transformer):
    """Implementation of a binary class classifier as scikit-learn `Pipeline` step."""

    def __init__(self, estimator: SKLearnPipelineModule):
        self.estimator = estimator

    def fit(self, X: Sequence[Any], y: Sequence[Any] | None = None) -> Transformer:
        self.estimator.fit(X, y)
        return self

    def transform(self, X: Sequence[Any]) -> Sequence[Any]:
        return self.estimator.predict_proba(X)[:, 1]


class FunctionTransformer(Transformer):
    """Implementation of a transformer function as scikit-learn `Pipeline` step."""

    def __init__(self, func: Callable):
        self.func = func

    def fit(self, X: Sequence[Any], y: Sequence[Any] | None = None) -> Transformer:
        return self

    def transform(self, X: Sequence[Any]) -> Sequence[Any]:
        return self.func(X)


class Tee(AdvancedTransformer):
    """Implementation of a custom transformer allowing to perform two separate tasks on a given input."""

    def __init__(self, transformers: list[Transformer]):
        super().__init__()
        self.transformers = transformers

    def set_value(self, key: str, value: Any) -> AdvancedTransformer:
        super().set_value(key, value)
        for transformer in self.transformers:
            if isinstance(transformer, AdvancedTransformer):
                transformer.set_value(key, value)

        return self

    def fit(self, X: Sequence[Any], y: Sequence[Any] | None = None) -> Transformer:
        for transformer in self.transformers:
            transformer.fit(X, y)

        return self

    def transform(self, X: Any) -> Any:
        for transformer in self.transformers:
            transformer.transform(X)

        return X


class TransformerWrapper(AdvancedTransformer):
    """Base class for a transformer wrapper.

    This class is derived from `AdvancedTransformer` and has a default implementation of all functions
    by forwarding the call to the wrapped transformer. A sub class may add or change functionality
    by overriding functions.
    """

    def __init__(self, wrapped_transformer: Transformer):
        super().__init__()
        self.wrapped_transformer = wrapped_transformer

    def set_value(self, key: str, value: Any | None) -> AdvancedTransformer:
        super().set_value(key, value)
        if isinstance(self.wrapped_transformer, AdvancedTransformer):
            self.wrapped_transformer.set_value(key, value)
        return self

    def fit(self, X: Sequence[Any], y: Sequence[Any] | None = None) -> Transformer:
        self.wrapped_transformer.fit(X, y)
        return self

    def transform(self, X: Any) -> Any:
        return self.wrapped_transformer.transform(X)


class NumpyTransformer(TransformerWrapper):
    """Implementation of a transformer wrapper to handle writing numpy data to CSV."""

    def __init__(self, transformer: Transformer, header: list[str] | None, path: Path):
        super().__init__(transformer)
        self.csv_writer = CsvWriter(header=header, path=path)

    def set_value(self, key: str, value: Any | None) -> AdvancedTransformer:
        super().set_value(key, value)
        self.csv_writer.set_value(key, value)
        return self

    def transform(self, X: Any) -> Any:
        result = super().transform(X)
        self.csv_writer.transform(result)
        return result


class CsvWriter(AdvancedTransformer):
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

    def _join_reader(self, filename: str) -> Iterator:
        with open(self.path.parent / filename, "r") as f:
            reader = csv.reader(f)
            yield from reader

    def _write_rows(self, writer: DataWriter, rows: Any, write_header: bool) -> None:
        all_headers = []
        all_data = []

        if self.include_labels:
            labels = self._values.get(TRANSFORMER_LABELS_KEY, None)
            if labels is not None:
                all_headers.append("label")
                all_data.append(labels.reshape(-1, 1))

        if self.include_meta:
            meta = self._values.get(TRANSFORMER_META_KEY, None)
            if meta is not None:
                if len(meta.shape) < 2:
                    meta = meta.reshape(meta.shape[0], -1)
                all_headers.extend([f"{self.header_prefix}.meta{n}" for n in range(meta.shape[1])])
                all_data.append(meta)

        if self.include_batch:
            all_headers.append("batch")
            all_data.append(np.ones((rows.shape[0], 1)) * self.n_batches)

        if self.include_input:
            if self.header is None:
                all_headers.extend([f"{self.header_prefix}{i}" for i in range(len(rows[0]))])
            else:
                all_headers.extend(self.header)
            all_data.append(rows)

        if self.input_files is not None:
            for filename in self.input_files:
                rows = self._join_reader(filename)
                all_headers.extend(next(rows))
                all_data.append(rows)

        if write_header:
            writer.writerow(all_headers)
        for row in zip(*all_data):
            writer.writerow(chain(*row))

    def transform(self, features: np.ndarray) -> Any:
        """Write numpy feature vector to CSV output file."""
        if TRANSFORMER_PHASE_KEY not in self._values or self._values[TRANSFORMER_PHASE_KEY] != self.write_on_phase:
            return features

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

            if isinstance(features, np.ndarray):
                if len(features.shape) == 1:
                    values = features.reshape(-1, 1)
                elif len(features.shape) == 2:
                    values = features
                else:
                    raise ValueError(f"{__name__}: unsupported shape: {features.shape}")
                self._write_rows(writer, values, write_header)
            else:
                for row in features:
                    writer.writerow(row)

        self.n_batches += 1

        return features
