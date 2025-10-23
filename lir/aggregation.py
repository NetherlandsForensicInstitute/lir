import csv
from abc import abstractmethod, ABC
from pathlib import Path
from typing import Any, IO
from collections import OrderedDict
from collections.abc import Mapping, Callable

import numpy as np

from lir import compat


class Aggregation(ABC):
    @abstractmethod
    def report(self, llrs: np.ndarray, labels: np.ndarray | None, parameters: dict[str, Any]) -> None:
        """
        Report that new results are available.

        :param llrs: log-LR values
        :param labels: corresponding labels
        :param parameters: parameters that identify the system producing the results
        """
        raise NotImplementedError

    def close(self) -> None:
        """
        Finalize the aggregation; no more results will come in.

        The close method is called at the end of gathering the aggregation(s) to ensure files are closed, buffers are
        cleared, or other things that need to finish / tear down.
        """
        pass


class WriteMetricsToCsv(Aggregation):
    def __init__(self, path: Path, metrics: Mapping[str, Callable]):
        self.path = path
        self._file: IO[Any] | None = None
        self._writer: csv.DictWriter | None = None
        self.metrics = metrics

    def report(self, llrs: np.ndarray, labels: np.ndarray | None, parameters: dict[str, Any]) -> None:
        metrics = [(key, metric(compat.llr_to_lr(llrs), labels)) for key, metric in self.metrics.items()]
        results = OrderedDict(list(parameters.items()) + metrics)

        # Record column header names only once to the CSV
        if self._writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._file = open(self.path, "w")
            self._writer = csv.DictWriter(self._file, fieldnames=results.keys())
            self._writer.writeheader()

        self._writer.writerow(results)
        self._file.flush()  # type: ignore

    def close(self) -> None:
        if self._file:
            self._file.close()
