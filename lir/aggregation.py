import csv
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Callable, Mapping
from functools import partial
from pathlib import Path
from typing import IO, Any, NamedTuple

from matplotlib import pyplot as plt

from lir.algorithms.bayeserror import plot_nbe as nbe
from lir.config.base import ContextAwareDict, config_parser
from lir.data.models import LLRData
from lir.lrsystems.lrsystems import LRSystem
from lir.plotting import calibrator_fit, llr_interval, lr_histogram, pav, tippett
from lir.plotting.expected_calibration_error import plot_ece as ece


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

    def __init__(self, plot_fn: Callable, output_dir: Path | None = None) -> None:
        self.output_path = output_dir
        self.plot_fn = partial(plot_fn)

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


@config_parser
def plot_pav(config: ContextAwareDict, output_dir: Path) -> AggregatePlot:
    return AggregatePlot(output_dir=output_dir, plot_fn=pav)


@config_parser
def plot_ece(config: ContextAwareDict, output_dir: Path) -> AggregatePlot:
    return AggregatePlot(output_dir=output_dir, plot_fn=ece)


@config_parser
def plot_lr_histogram(config: ContextAwareDict, output_dir: Path) -> AggregatePlot:
    return AggregatePlot(output_dir=output_dir, plot_fn=lr_histogram)


@config_parser
def plot_llr_interval(config: ContextAwareDict, output_dir: Path) -> AggregatePlot:
    return AggregatePlot(output_dir=output_dir, plot_fn=llr_interval)


@config_parser
def plot_nbe(config: ContextAwareDict, output_dir: Path) -> AggregatePlot:
    return AggregatePlot(output_dir=output_dir, plot_fn=nbe)


@config_parser
def plot_tipett(config: ContextAwareDict, output_dir: Path) -> AggregatePlot:
    return AggregatePlot(output_dir=output_dir, plot_fn=tippett)


@config_parser
def plot_calibrator_fit(config: ContextAwareDict, output_dir: Path) -> AggregatePlot:
    return AggregatePlot(output_dir=output_dir, plot_fn=calibrator_fit)


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
