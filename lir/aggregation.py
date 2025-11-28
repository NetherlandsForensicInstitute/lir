import csv
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Callable, Mapping
from functools import partial
from pathlib import Path
from typing import IO, Any, NamedTuple

from matplotlib import pyplot as plt

from lir.data.models import LLRData
from lir.lrsystems.lrsystems import LRSystem
from lir.plotting import llr_interval, lr_histogram, pav
from lir.plotting.expected_calibration_error import plot_ece


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

    output_path: Path | None = None
    plot_fn: Callable

    def __init__(self, output_dir: str | None = None) -> None:
        if output_dir:
            self.output_path = Path(output_dir)

    def report(self, data: AggregationData) -> None:
        """Plot the data when new results are available."""
        fig, ax = plt.subplots()

        llrdata = data.llrdata
        parameters = data.parameters

        self.plot_fn(llrdata=llrdata, ax=ax)

        # Only save the figure when an output path is provided.
        if self.output_path is not None:
            dir_name = self.output_path
            param_string = '__'.join(f'{k}={v}' for k, v in parameters.items())
            file_name = dir_name / param_string / f'{self.__class__.__name__}.png'
            dir_name.mkdir(exist_ok=True, parents=True)

            fig.savefig(file_name)


class PAVPlot(AggregatePlot):
    plot_fn = partial(pav)


class ECEPlot(AggregatePlot):
    plot_fn = partial(plot_ece)


class LRHistogramPlot(AggregatePlot):
    plot_fn = partial(lr_histogram)


class LLRIntervalPlot(AggregatePlot):
    plot_fn = partial(llr_interval)


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
