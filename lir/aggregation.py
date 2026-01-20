import csv
import logging
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import IO, Any, NamedTuple

from matplotlib import pyplot as plt

from lir.algorithms.bayeserror import plot_nbe as nbe
from lir.algorithms.invariance_bounds import plot_invariance_delta_functions as invariance_delta_functions
from lir.algorithms.llr_overestimation import plot_llr_overestimation as llr_overestimation
from lir.config.base import ContextAwareDict, YamlParseError, config_parser, pop_field
from lir.config.metrics import parse_individual_metric
from lir.data.models import LLRData
from lir.lrsystems.lrsystems import LRSystem
from lir.plotting import llr_interval, lr_histogram, pav, tippett
from lir.plotting.expected_calibration_error import plot_ece as ece


LOG = logging.getLogger(__name__)


class AggregationData(NamedTuple):
    """
    Fields:
    - llrdata: the LLR data containing LLRs and labels.
    - lrsystem: the model that produced the results
    - parameters: parameters that identify the system producing the results.
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

    def __init__(self, plot_fn: Callable, plot_name: str, output_dir: Path | None = None, **kwargs: Any) -> None:
        self.output_path = output_dir
        self.plot_fn = plot_fn
        self.plot_name = plot_name
        self.plot_fn_args = kwargs

    def report(self, data: AggregationData) -> None:
        """Plot the data when new results are available."""
        fig, ax = plt.subplots()

        llrdata = data.llrdata
        parameters = data.parameters

        try:
            self.plot_fn(llrdata=llrdata, ax=ax, **self.plot_fn_args)
        except ValueError as e:
            LOG.warning(f'Could not generate plot {self.plot_name} for parameters {parameters}: {e}')
            return

        # Only save the figure when an output path is provided.
        if self.output_path is not None:
            dir_name = self.output_path
            param_string = '__'.join(f'{k}={v}' for k, v in parameters.items()) + '_' if parameters else ''
            plot_arguments = (
                '_' + '_'.join(f'{k}={v}' for k, v in self.plot_fn_args.items()) if self.plot_fn_args else ''
            )

            file_name = dir_name / f'{param_string}{self.plot_name}{plot_arguments}.png'
            dir_name.mkdir(exist_ok=True, parents=True)

            LOG.info(f'Saving plot {self.plot_name} for parameters {parameters} to {file_name}')
            fig.savefig(file_name)


@config_parser
def plot_pav(config: ContextAwareDict, output_dir: Path) -> AggregatePlot:
    plot_name = pop_field(config, 'plot_name', default='PAV')
    return AggregatePlot(pav, plot_name, output_dir, **config)


@config_parser
def plot_ece(config: ContextAwareDict, output_dir: Path) -> AggregatePlot:
    plot_name = pop_field(config, 'plot_name', default='ECE')
    return AggregatePlot(ece, plot_name, output_dir, **config)


@config_parser
def plot_lr_histogram(config: ContextAwareDict, output_dir: Path) -> AggregatePlot:
    plot_name = pop_field(config, 'plot_name', default='LR_Histogram')
    return AggregatePlot(lr_histogram, plot_name, output_dir, **config)


@config_parser
def plot_llr_interval(config: ContextAwareDict, output_dir: Path) -> AggregatePlot:
    plot_name = pop_field(config, 'plot_name', default='LLR_Interval')
    return AggregatePlot(llr_interval, plot_name, output_dir, **config)


@config_parser
def plot_llr_overestimation(config: ContextAwareDict, output_dir: Path) -> AggregatePlot:
    plot_name = pop_field(config, 'plot_name', default='LLR_Overestimation')
    return AggregatePlot(llr_overestimation, plot_name, output_dir, **config)


@config_parser
def plot_nbe(config: ContextAwareDict, output_dir: Path) -> AggregatePlot:
    plot_name = pop_field(config, 'plot_name', default='NBE')
    return AggregatePlot(nbe, plot_name, output_dir, **config)


@config_parser
def plot_tippett(config: ContextAwareDict, output_dir: Path) -> AggregatePlot:
    plot_name = pop_field(config, 'plot_name', default='Tippett')
    return AggregatePlot(tippett, plot_name, output_dir, **config)


@config_parser
def plot_invariance_delta_function(config: ContextAwareDict, output_dir: Path) -> AggregatePlot:
    plot_name = pop_field(config, 'plot_name', default='Invariance_Delta_Functions')
    return AggregatePlot(invariance_delta_functions, plot_name, output_dir, **config)


class WriteMetricsToCsv(Aggregation):
    def __init__(self, output_dir: Path, columns: Mapping[str, Callable]):
        self.path = output_dir / 'metrics.csv'
        self._file: IO[Any] | None = None
        self._writer: csv.DictWriter | None = None
        self.columns = columns

    def report(self, data: AggregationData) -> None:
        columns = [(key, metric(data.llrdata)) for key, metric in self.columns.items()]
        metrics = []
        for name, value in columns:
            if isinstance(value, (list, tuple)):
                for index, metric_value in enumerate(value):
                    metrics.append((f'{name}_{index}', str(metric_value)))
            else:
                metrics.append((name, str(value)))

        results = OrderedDict(list(data.parameters.items()) + metrics)

        # Record column header names only once to the CSV
        if self._writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._file = open(self.path, 'w', newline='')  # noqa: SIM115
            self._writer = csv.DictWriter(self._file, fieldnames=results.keys())
            self._writer.writeheader()
        self._writer.writerow(results)
        self._file.flush()  # type: ignore

    def close(self) -> None:
        if self._file:
            self._file.close()


@config_parser
def metrics_csv(config: ContextAwareDict, output_dir: Path) -> WriteMetricsToCsv:
    columns = pop_field(config, 'columns', default=['cllr', 'cllr_min'])
    if not isinstance(columns, Sequence):
        raise YamlParseError(
            config.context,
            'Invalid metrics configuration; expected a list of metric names.',
        )

    columns = {name: parse_individual_metric(name, output_dir, config.context) for name in columns}
    return WriteMetricsToCsv(output_dir, columns)
