import csv
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import IO, Any, NamedTuple

from matplotlib import pyplot as plt

from lir.data.models import LLRData
from lir.lrsystems.lrsystems import LRSystem
from lir.plotting import Canvas


class AggregationData(NamedTuple):
    """
    Fields:
    - llrdata: the LLR data containing LLRs and labels.
    - lrsystem: the model that produced the results
    - parameters: parameters that identify the system producing the results
    """

    llrdata: LLRData
    lrsystem: LRSystem
    parameters: dict[str, Any]


class Aggregation(ABC):
    @abstractmethod
    def report(self, data: AggregationData) -> None:
        """
        Report that new results are available.

        :param data: a named tuple containing the results
        """
        raise NotImplementedError

    def close(self) -> None:  # noqa: B027
        """
        Finalize the aggregation; no more results will come in.

        The close method is called at the end of gathering the aggregation(s) to ensure files are closed, buffers are
        cleared, or other things that need to finish / tear down.
        """
        pass


class AggregatePlot(Aggregation):
    """Aggregation that generates plots by repeatedly calling a plotting function."""

    def __init__(self, plot_function: Callable, output_dir: str) -> None:
        super().__init__()

        self.f = plot_function
        self.dir = output_dir
        self.plot_type = plot_function.__name__
        self._fig, self._ax = plt.subplots(figsize=(10, 8))
        self._canvas = Canvas(self._ax)

    def report(self, data: AggregationData) -> None:
        self._canvas.plot(
            [],
            [],
            marker='None',
            linestyle='None',
            color='white',  # This is necessary to avoid matplotlib from cycling through colours
            label=', '.join(f'{k}={v}' for k, v in data.parameters.items()),
        )  # Dummy plot to add legend entry

        self.f(None, data.llrdata, self._canvas)

    def close(self) -> None:
        """Generate and save each plot after all results have been reported."""
        self._ax.set_title(f'Aggregated {self.plot_type}')
        self._fig.savefig(f'{self.dir}/aggregated_{self.plot_type}.png')


class WriteMetricsToCsv(Aggregation):
    def __init__(self, output_dir: Path, metrics: Mapping[str, Callable]):
        self.path = output_dir / 'metrics.csv'
        self._file: IO[Any] | None = None
        self._writer: csv.DictWriter | None = None
        self.metrics = metrics

    def report(self, data: AggregationData) -> None:
        metrics = [(key, metric(data.llrdata.llrs, data.llrdata.labels)) for key, metric in self.metrics.items()]
        results = OrderedDict(list(data.parameters.items()) + metrics)

        # Record column header names only once to the CSV
        if self._writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._file = open(self.path, 'w')  # noqa: SIM115
            self._writer = csv.DictWriter(self._file, fieldnames=results.keys())
            self._writer.writeheader()
        self._writer.writerow(results)
        self._file.flush()  # type: ignore

    def close(self) -> None:
        if self._file:
            self._file.close()
